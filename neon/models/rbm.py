# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple restricted Boltzmann Machine model.
"""

import logging
from neon.models.model import Model
from neon.util.compat import range

logger = logging.getLogger(__name__)


class RBM(Model):

    """
    Restricted Boltzmann Machine with binary visible and binary hidden units
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for req_param in ['layers', 'batch_size']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)
        self.cost.initialize(kwargs)

    def fit(self, dataset):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))
        inputs = dataset.get_inputs(train=True)['train']
        nin = self.layers[0].nin
        self.nlayers = len(self.layers)
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        self.temp = self.backend.empty((nin, self.batch_size), self.temp_dtype)

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = len(inputs)
        logger.info('commencing model fitting')
        for epoch in range(self.num_epochs):
            error = 0.0
            for batch in range(num_batches):
                inputs_batch = dataset.get_batch(inputs, batch)
                self.positive(inputs_batch)
                self.negative(inputs_batch)
                error += self.cost.apply_function(inputs_batch)
                self.update(epoch)
            logger.info('epoch: %d, total training error: %0.5f', epoch,
                        error / num_batches)

    def positive(self, inputs):
        """Wrapper for RBMLayer.positive"""
        self.layers[0].positive(inputs)
        return None

    def negative(self, inputs):
        """Wrapper for RBMLayer.negative"""
        self.layers[0].negative(inputs)
        return None

    def update(self, epoch):
        """Wrapper for RBMLayer.update"""
        self.layers[0].update(epoch)
