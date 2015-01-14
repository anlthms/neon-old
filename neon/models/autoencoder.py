# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train stacked autoencoder models and run inference.
"""

import logging

from neon.models.mlp import MLP
from neon.util.compat import range

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
            logger.info("%s", str(layer))
        ds = datasets[0]
        inputs = ds.get_inputs(train=True)['train']
        targets = ds.get_inputs(train=True)['train']

        num_batches = len(inputs)
        logger.info('commencing model fitting')
        error = self.backend.empty((1, 1))
        for epoch in range(self.num_epochs):
            error.fill(0.0)
            for batch in range(num_batches):
                inputs_batch = ds.get_batch(inputs, batch)
                targets_batch = ds.get_batch(targets, batch)
                self.fprop(inputs_batch)
                self.bprop(targets_batch, inputs_batch)
                self.backend.add(error,
                                 self.cost.apply_function(targets_batch),
                                 error)
                self.update(epoch)
            logger.info('epoch: %d, total training error: %0.5f',
                        epoch, error.asnumpyarray() / num_batches)
