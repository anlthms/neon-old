"""
Contains code to train Google Brain models and run inference.
"""

import logging
import math

from mylearn.models.layer import LocalFilteringLayer
from mylearn.models.mlp import MLP

logger = logging.getLogger(__name__)


class GB(MLP):
    """
    Google Brain class
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def pretrain(self, datasets):
        logger.info('commencing unsupervised pretraining')
        inputs = datasets[0].get_inputs(train=True)['train']
        nrecs, nin = inputs.shape
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        self.trainable_layers = []
        self.temp = []
        for ind in xrange(len(self.layers)):
            layer = self.layers[ind]
            if isinstance(layer, LocalFilteringLayer):
                self.trainable_layers.append(ind)
                self.temp.append(self.backend.zeros((self.batch_size, nin)))
            nin = layer.nout

        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for ind in xrange(len(self.trainable_layers)):
            layer = self.layers[self.trainable_layers[ind]]
            for epoch in xrange(self.num_pretraining_epochs):
                error = 0.0
                for batch in xrange(num_batches):
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, nrecs)
                    output = inputs[start_idx:end_idx]
                    # Forward propagate the input all the way to
                    # the layer that we are pretraining.
                    for i in xrange(self.trainable_layers[ind]):
                        self.layers[i].fprop(output)
                        output = self.layers[i].output

                    layer.pretrain(output)
                    error += self.pretraining_cost.apply_function(
                        self.backend, layer.recon,
                        output, self.temp[ind])
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / num_batches))

    def train(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing supervised training')
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
        nrecs, nin = inputs.shape
        tempbuf = self.backend.zeros((self.batch_size, targets.shape[1]))
        self.temp = [tempbuf, tempbuf.copy()]

        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.fprop(inputs[start_idx:end_idx])
                self.bprop(targets[start_idx:end_idx],
                           inputs[start_idx:end_idx],
                           epoch, self.momentum)
                error += self.cost.apply_function(self.backend,
                                                  self.layers[-1].output,
                                                  targets[start_idx:end_idx],
                                                  self.temp)
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))

    def fit(self, datasets):
        self.pretrain(datasets)
        self.train(datasets)
