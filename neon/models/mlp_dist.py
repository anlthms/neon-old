# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple multi-layer perceptron model.
"""

import logging
import math

from neon.models.mlp import MLP
from neon.models.layer import LayerWithNoBiasDist, LayerDist
from neon.util.compat import MPI_INSTALLED

logger = logging.getLogger(__name__)

if MPI_INSTALLED:
    from mpi4py import MPI
else:
    logger.error('mpi4py not found')


class MLPDist(MLP):

    """
    MPI distributed fully connected, feed-forward, multi-layer perceptron model
    """

    def adjust_for_dist(self):
        # MPI: call adjust_for_dist for each layer
        for i in xrange(0, self.nlayers):
            layer = self.layers[i]
            layer_no_bias_dist = isinstance(layer, LayerWithNoBiasDist)
            layer_dist = isinstance(layer, LayerDist)
            if layer_no_bias_dist or layer_dist:
                # fully connected layer: no halo transfers needed
                # layer.nout_ stores the non-dist layer.nout value
                layer.nout_ = layer.nout
                if i < self.nlayers - 1:
                    # overwrite layer.nout with dist value
                    if layer.nout % self.comm.size != 0:
                        raise ValueError('Unsupported layer.nout % '
                                         'self.comm.size != 0')
                    layer.nout = layer.nout / self.comm.size
                    # when non-squared comm sizes are allowed
                    # layer.nout = (layer.nout // self.comm.size +
                    #     (layer.nout % self.comm.size >
                    #        self.comm.rank))
                prev_layer_no_bias_dist = (
                    isinstance(self.layers[i - 1], LayerWithNoBiasDist))
                prev_layer_dist = isinstance(self.layers[i - 1], LayerDist)
                if i == 0 or prev_layer_no_bias_dist or prev_layer_dist:
                    # split the inputs nin across self.comm.size
                    start_idx = 0
                    nin = layer.nin
                    if layer_dist:
                        nin -= 1
                    for j in range(self.comm.rank):
                        start_idx += (nin // self.comm.size +
                                      (nin % self.comm.size > j))
                    layer.nin = (nin // self.comm.size +
                                 (nin % self.comm.size > self.comm.rank))
                    layer.in_indices = range(start_idx, start_idx + layer.nin)
                    layer.out_indices = layer.in_indices
                    is_last_rank = (self.comm.rank == self.comm.size-1)
                    if layer_dist and is_last_rank:
                        # add the bias term for the last rank process
                        layer.in_indices = range(start_idx,
                                                 start_idx + layer.nin + 1)
                        layer.nin += 1
                    if prev_layer_no_bias_dist:
                        layer.prev_layer = 'LayerWithNoBiasDist'
                    elif prev_layer_dist:
                        layer.prev_layer = 'LayerDist'
                else:
                    raise ValueError('Unsupported previous layer for '
                                     'LayerWithNoBiasDist or LayerDist')
            layer.adjust_for_dist()

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.debug("%s" % str(layer))
        self.comm = MPI.COMM_WORLD
        self.adjust_for_dist()
        ds = datasets[0]
        inputs = ds.get_inputs(train=True)['train']
        targets = ds.get_targets(train=True)['train']
        nrecs = inputs.shape[0]
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        tempbuf = self.backend.zeros((self.layers[-1].nout, self.batch_size),
                                     self.temp_dtype)
        self.temp = [tempbuf, tempbuf.copy()]

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        logger.info('commencing model fitting')
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(inputs.nbatches):
                if self.comm.rank == 0:
                    logger.debug('batch = %d' % (batch))
                inputs_batch = ds.get_batch(inputs, batch)
                targets_batch = ds.get_batch(targets, batch)
                self.fprop(inputs_batch)
                self.bprop(targets_batch, inputs_batch, epoch)
                if self.comm.rank == 0:
                    error += self.cost.apply_function(self.backend,
                                                      self.layers[-1].output,
                                                      targets_batch,
                                                      self.temp)
            if self.comm.rank == 0:
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / inputs.nbatches))
            for layer in self.layers:
                logger.debug("%s", layer)

    def predict_set(self, ds, inputs, predsdict, setname):
        if self.comm.rank == 0:
            preds = self.backend.empty((inputs[setname].nbatches,
                                       self.batch_size))
            predsdict[setname] = preds

        for batch in xrange(inputs[setname].nbatches):
            inputs_batch = ds.get_batch(inputs[setname], batch)
            self.fprop(inputs_batch)
            if self.comm.rank == 0:
                outputs = self.layers[-1].output
                self.backend.argmax(outputs, axis=0,
                                    out=preds[batch:(batch+1)])

    def predict(self, datasets, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given datasets.
        """
        res = []
        for ds in datasets:
            inputs = ds.get_inputs(train, test, validation)
            predsdict = dict()
            if train and 'train' in inputs:
                self.predict_set(ds, inputs, predsdict, 'train')
            if test and 'test' in inputs:
                self.predict_set(ds, inputs, predsdict, 'test')
            if validation and 'validation' in inputs:
                self.predict_set(ds, inputs, predsdict, 'validation')
            if self.comm.rank == 0:
                if len(predsdict) is 0:
                    logger.error(
                        "must specify >=1 of: train, test, validation")
                res.append(predsdict)
        return res

    def fprop(self, inputs):
        # call MLP's fprop: doesn't work for FC->FC connections
        # super(ConvnetDist, self).fprop(inputs)
        # handle FC-> FC connections
        y = inputs
        for layer in self.layers:
            if layer.pos > 0:
                y = y.take(layer.out_indices, axis=0)
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs, epoch):
        i = self.nlayers - 1
        lastlayer = self.layers[i]

        error = self.backend.zeros((self.layers[-1].nout, self.batch_size))
        # apply derivative on root node's FC layer output
        if self.comm.rank == 0:
            error = self.cost.apply_derivative(self.backend,
                                               lastlayer.output, targets,
                                               self.temp)
            self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                                out=error)
        error._tensor = self.comm.bcast(error.raw())
        # Update the output layer.
        lastlayer.pre_act_ = lastlayer.pre_act
        prev_layer_no_bias_dist = (
            isinstance(self.layers[i - 1], LayerWithNoBiasDist))
        prev_layer_dist = isinstance(self.layers[i - 1], LayerDist)
        while i > 0:
            prev_layer_no_bias_dist = (
                isinstance(self.layers[i - 1], LayerWithNoBiasDist))
            prev_layer_dist = isinstance(self.layers[i - 1], LayerDist)
            if prev_layer_dist or prev_layer_no_bias_dist:
                self.layers[i].bprop(error,
                                     self.layers[i - 1].output.
                                     take(self.layers[i].out_indices, axis=0),
                                     epoch)
            else:
                self.layers[i].bprop(error,
                                     self.layers[i - 1].output,
                                     epoch)
            error = self.layers[i].berror
            i -= 1
            # extract self.layers[i].pre_act terms
            self.layers[i].pre_act_ = self.layers[i].pre_act.take(
                self.layers[i + 1].out_indices, axis=0)

        # first FC layer
        self.layers[i].bprop(self.layers[i + 1].berror,
                             inputs,
                             epoch)
