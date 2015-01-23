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
            self.backend.begin()
            logger.info("%s", str(layer))
            self.backend.end()
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
        error = self.backend.empty((1, 1))
        while self.epochs_complete < self.num_epochs:
            self.backend.begin()
            error.fill(0.0)
            for batch in range(num_batches):
                self.backend.begin()
                inputs_batch = dataset.get_batch(inputs, batch)
                self.positive(inputs_batch)
                self.negative(inputs_batch)
                self.backend.add(error, self.cost.apply_function(inputs_batch),
                                 error)
                self.update(self.epochs_complete)
                self.backend.end()
            logger.info('epoch: %d, total training error: %0.5f',
                        self.epochs_complete,
                        error.asnumpyarray() / num_batches)
            self.epochs_complete += 1
            self.backend.end()

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
