"""
Simple restricted Boltzmann Machine model.
"""

import logging
import math
from mylearn.models.model import Model

logger = logging.getLogger(__name__)


class RBM(Model):

    """
    Restricted Boltzmann Machine with binary visible and binary hidden units
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for req_param in ['layers']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.info("%s" % str(layer))
        inputs = datasets[0].get_inputs(train=True)['train']
        nrecs = inputs.shape[0]
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        logger.info('commencing model fitting')
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.positive(inputs[start_idx:end_idx])
                self.negative(inputs[start_idx:end_idx])
                self.update(self.learning_rate, epoch, self.momentum)

                error += self.cost.apply_function(inputs[start_idx:end_idx],
                                                  self.layers[0].x_minus[:,
                                                  0:(self.layers[0].x_minus.
                                                     shape[1] - 1)])
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))

    def positive(self, inputs):
        """Wrapper for RBMLayer.positive"""
        self.layers[0].positive(inputs)
        return None

    def negative(self, inputs):
        """Wrapper for RBMLayer.negative"""
        self.layers[0].negative(inputs)
        return None

    def update(self, epsilon, epoch, momentum):
        """Wrapper for RBMLayer.update"""
        self.layers[0].update(epsilon, epoch, momentum)
