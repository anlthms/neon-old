# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train stacked autoencoder models and run inference.
"""

import logging
import math

from neon.models.mlp import MLP

logger = logging.getLogger(__name__)


class Autoencoder(MLP):
    """
    Adaptation of multi-layer perceptron.
    """

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.info("%s" % str(layer))
        ds = datasets[0]
        inputs = ds.get_inputs(train=True)['train']
        targets = ds.get_inputs(train=True)['train']
        assert 'batch_size' in self.__dict__

        logger.info('commencing model fitting')
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(inputs.nbatches):
                inputs_batch = ds.get_batch(inputs, batch)
                targets_batch = ds.get_batch(targets, batch)
                self.fprop(inputs_batch)
                self.bprop(targets_batch, inputs_batch, epoch)
                error += self.cost.apply_function(
                    self.backend, self.layers[-1].output,
                    targets_batch,
                    self.temp)
            logger.info('epoch: %d, total training error: %0.5f',
                        epoch, error / inputs.nbatches)
