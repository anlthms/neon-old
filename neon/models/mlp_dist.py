"""
Simple multi-layer perceptron model.
"""

import logging
import math

from neon.models.mlp import MLP
from neon.models.layer import LayerWithNoBiasDist
from neon.util.compat import MPI_INSTALLED

logger = logging.getLogger(__name__)

if MPI_INSTALLED:
    from mpi4py import MPI
else:
    logger.error('mpi4py not found')


class MLPDist(MLP):

    """
    Fully connected, feed-forward, multi-layer perceptron model
    """

    def adjust_for_dist(self):
        # MPI: call adjust_for_dist for each layer
        for i in xrange(0, self.nlayers):
            layer = self.layers[i]
            if isinstance(layer, LayerWithNoBiasDist):
                # fully connected layer: no halo transfers needed
                layer.nout_ = layer.nout
                if i < self.nlayers - 1:
                    if layer.nout % MPI.COMM_WORLD.size != 0:
                        raise ValueError('Unsupported layer.nout % '
                                         'MPI.COMM_WORLD.size != 0')
                    layer.nout = layer.nout / MPI.COMM_WORLD.size
                is_prev_layer_with_no_bias_dist = (
                    isinstance(self.layers[i - 1], LayerWithNoBiasDist))
                if i == 0 or is_prev_layer_with_no_bias_dist:
                    # split the inputs nin across MPI.COMM_WORLD.size
                    if layer.nin % MPI.COMM_WORLD.size != 0:
                        raise ValueError('Unsupported layer.nin % '
                                         'MPI.COMM_WORLD.size != 0')
                    layer.nin = layer.nin / MPI.COMM_WORLD.size
                    layer.out_indices = range(MPI.COMM_WORLD.rank * layer.nin,
                                              (MPI.COMM_WORLD.rank + 1) *
                                              layer.nin)
                    layer.prev_layer = 'LayerWithNoBiasDist'
                else:
                    raise ValueError('Unsupported previous layer for '
                                     'LayerWithNoBiasDist')
            layer.adjust_for_dist()

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        # for layer in self.layers:
        #    logger.info("%s" % str(layer))
        self.adjust_for_dist()
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
        nrecs = inputs.shape[0]
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        tempbuf = self.backend.zeros((self.batch_size, self.layers[-1].nout),
                                     self.temp_dtype)
        self.temp = [tempbuf, tempbuf.copy()]

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        logger.info('commencing model fitting')
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                # if MPI.COMM_WORLD.rank == 0:
                #    logger.info('batch = %d' % (batch))
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.fprop(inputs[start_idx:end_idx])
                self.bprop(targets[start_idx:end_idx],
                           inputs[start_idx:end_idx],
                           epoch, self.momentum)
                if MPI.COMM_WORLD.rank == 0:
                    error += self.cost.apply_function(self.backend,
                                                      self.layers[-1].output,
                                                      targets[
                                                          start_idx:end_idx],
                                                      self.temp)
            if MPI.COMM_WORLD.rank == 0:
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / num_batches))
            for layer in self.layers:
                logger.debug("%s", layer)

    def predict_set(self, inputs):
        nrecs = inputs.shape[0]
        if MPI.COMM_WORLD.rank == 0:
            self.outputs = self.backend.zeros((nrecs, self.layers[-1].nout))
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for batch in xrange(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, nrecs)
            self.fprop(inputs[start_idx:end_idx])
            if MPI.COMM_WORLD.rank == 0:
                self.outputs[start_idx:end_idx, :] = self.layers[-1].output

    def predict(self, datasets, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given datasets.
        """
        res = []
        for dataset in datasets:
            inputs = dataset.get_inputs(train, test, validation)
            preds = dict()
            if train and 'train' in inputs:
                self.predict_set(inputs['train'])
                if MPI.COMM_WORLD.rank == 0:
                    preds['train'] = dataset.backend.argmax(
                        self.outputs, axis=1)
            if test and 'test' in inputs:
                self.predict_set(inputs['test'])
                if MPI.COMM_WORLD.rank == 0:
                    preds['test'] = dataset.backend.argmax(
                        self.outputs, axis=1)
            if validation and 'validation' in inputs:
                self.predict_set(inputs['validation'])
                if MPI.COMM_WORLD.rank == 0:
                    preds['validation'] = dataset.backend.argmax(
                        self.outputs, axis=1)
            if MPI.COMM_WORLD.rank == 0:
                if len(preds) is 0:
                    logger.error(
                        "must specify >=1 of: train, test, validation")
                res.append(preds)

        return res

    def fprop(self, inputs):
        # call MLP's fprop: doesn't work for FC->FC connections
        # super(ConvnetDist, self).fprop(inputs)
        # handle FC-> FC connections
        y = inputs
        for layer in self.layers:
            # print MPI.COMM_WORLD.rank, layer.pos, y.shape
            if layer.pos > 0:
                y = y.take(layer.out_indices, axis=1)
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs, epoch, momentum):
        i = self.nlayers - 1
        lastlayer = self.layers[i]

        error = self.backend.zeros((self.batch_size, self.layers[-1].nout))
        # apply derivative on root node's FC layer output
        if MPI.COMM_WORLD.rank == 0:
            error = self.cost.apply_derivative(self.backend,
                                               lastlayer.output, targets,
                                               self.temp)
            self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                                out=error)
        error._tensor = MPI.COMM_WORLD.bcast(error.raw())
        # Update the output layer.
        lastlayer.pre_act_ = lastlayer.pre_act
        if isinstance(self.layers[i - 1], LayerWithNoBiasDist):
            lastlayer.bprop(error, self.layers[
                            i - 1].output.take(lastlayer.out_indices, axis=1),
                            epoch, momentum)
        else:
            lastlayer.bprop(error, self.layers[i - 1].output, epoch, momentum)
        i -= 1
        while i > 0 and isinstance(self.layers[i], LayerWithNoBiasDist):
            # extract self.layers[i].pre_act terms
            self.layers[i].pre_act_ = self.layers[i].pre_act.take(
                self.layers[i + 1].out_indices, axis=1)
            if isinstance(self.layers[i - 1], LayerWithNoBiasDist):
                self.layers[i].bprop(self.layers[i + 1].berror,
                                     self.layers[i - 1].output.
                                     take(self.layers[i].out_indices, axis=1),
                                     epoch, momentum)
            else:
                self.layers[i].bprop(self.layers[i + 1].berror,
                                     self.layers[i - 1].output,
                                     epoch, momentum)
            i -= 1

        # first FC layer
        self.layers[i].pre_act_ = self.layers[i].pre_act.take(
            self.layers[i + 1].out_indices, axis=1)
        self.layers[i].bprop(self.layers[i + 1].berror,
                             inputs,
                             epoch, momentum)
