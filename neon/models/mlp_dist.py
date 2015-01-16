# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple multi-layer perceptron model.
"""

import logging

from neon.models.mlp import MLP
from neon.models.layer import LayerDist
from neon.util.compat import MPI_INSTALLED, range

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
        for i in range(0, self.nlayers):
            layer = self.layers[i]
            layer_dist = isinstance(layer, LayerDist)
            if layer_dist:
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
                prev_layer_dist = isinstance(self.layers[i - 1], LayerDist)
                if i == 0 or prev_layer_dist:
                    # split the inputs nin across self.comm.size
                    start_idx = 0
                    nin = layer.nin
                    for j in range(self.comm.rank):
                        start_idx += (nin // self.comm.size +
                                      (nin % self.comm.size > j))
                    layer.nin = (nin // self.comm.size +
                                 (nin % self.comm.size > self.comm.rank))
                    layer.in_indices = range(start_idx, start_idx + layer.nin)
                    layer.out_indices = layer.in_indices
                    layer.prev_layer = 'LayerDist'
                else:
                    raise ValueError('Unsupported previous layer for '
                                     'LayerWithNoBiasDist or LayerDist')
            layer.adjust_for_dist()

    def fit(self, dataset):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.debug("%s", str(layer))
        self.comm = MPI.COMM_WORLD
        self.adjust_for_dist()
        ds = dataset
        inputs = ds.get_inputs(train=True)['train']
        targets = ds.get_targets(train=True)['train']

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        logger.info('commencing model fitting')
        error = self.backend.empty((1, 1))
        for epoch in range(self.num_epochs):
            error.fill(0)
            num_batches = len(inputs)
            for batch in range(num_batches):
                if self.comm.rank == 0:
                    logger.debug('batch = %d', batch)
                inputs_batch = ds.get_batch(inputs, batch)
                targets_batch = ds.get_batch(targets, batch)
                self.fprop(inputs_batch)
                self.bprop(targets_batch, inputs_batch)
                if self.comm.rank == 0:
                    self.backend.add(error,
                                     self.cost.apply_function(targets_batch),
                                     error)
                self.update(epoch)
            if self.comm.rank == 0:
                logger.info('epoch: %d, total training error: %0.5f', epoch,
                            error.asnumpyarray() / num_batches)
            for layer in self.layers:
                logger.debug("%s", layer)

    def predict_set(self, ds, inputs):
        for layer in self.layers:
            layer.set_train_mode(False)
        num_batches = len(inputs)
        nout = self.layers[-1].nout
        if self.comm.rank == 0:
            preds = []
        for batch in range(num_batches):
            inputs_batch = ds.get_batch(inputs, batch)
            preds_batch = self.backend.empty((nout, self.batch_size))
            self.fprop(inputs_batch)
            preds_batch[:] = self.get_classifier_output()
            preds.append(preds_batch)
        return preds

    def predict(self, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given dataset.
        """
        ds = self.dataset
        inputs = ds.get_inputs(train=train, test=test,
                               validation=validation)
        preds = dict()
        if train and 'train' in inputs:
            preds['train'] = self.predict_set(ds, inputs['train'])
        if test and 'test' in inputs:
            preds['test'] = self.predict_set(ds, inputs['test'])
        if validation and 'validation' in inputs:
            preds['validation'] = self.predict_set(ds,
                                                   inputs['validation'])
        if self.comm.rank == 0 and len(preds) is 0:
            logger.error("must specify >=1 of: train, test, validation")

        return preds

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

    def bprop(self, targets, inputs):
        i = self.nlayers - 1
        lastlayer = self.layers[i]

        error = self.backend.zeros((self.layers[-1].nout, self.batch_size))
        # apply derivative on root node's FC layer output
        if self.comm.rank == 0:
            error = self.cost.apply_derivative(targets)
            self.backend.divide(error, self.batch_size, out=error)
        error._tensor = self.comm.bcast(error.asnumpyarray())
        # Update the output layer.
        lastlayer.pre_act_ = lastlayer.pre_act
        prev_layer_dist = isinstance(self.layers[i - 1], LayerDist)
        while i > 0:
            prev_layer_dist = isinstance(self.layers[i - 1], LayerDist)
            if prev_layer_dist:
                self.layers[i].bprop(error, self.layers[i - 1].output.
                                     take(self.layers[i].out_indices, axis=0))
            else:
                self.layers[i].bprop(error, self.layers[i - 1].output)
            error = self.layers[i].deltas
            i -= 1
            # extract self.layers[i].pre_act terms
            self.layers[i].pre_act_ = self.layers[i].pre_act.take(
                self.layers[i + 1].out_indices, axis=0)

        # first FC layer
        self.layers[i].bprop(self.layers[i + 1].deltas, inputs)

    def update(self, epoch):
        for layer in self.layers:
            layer.update(epoch)
