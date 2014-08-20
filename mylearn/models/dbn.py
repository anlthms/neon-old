"""
Simple deep belief net.
"""

import logging
import math

from mylearn.models.layer import RBMLayer  # (u) created RBMLayer...
from mylearn.models.model import Model
from mylearn.util.factory import Factory

logger = logging.getLogger(__name__)


class DBN(Model):

    """
    deep belief net
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if isinstance(self.cost, str):
            self.cost = Factory.create(type=self.cost)

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing model fitting')
        inputs = datasets[0].get_inputs(train=True)['train']
        nrecs, nin = inputs.shape
        self.backend = datasets[0].backend
        self.backend.rng_init()
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        layers = []
        for i in xrange(self.nlayers):
            layer = self.lcreate(self.backend, nin, self.layers[i], i)
            logger.info('created layer:\n\t%s' % str(layer))
            layers.append(layer)
            nin = layer.nout - 1  # strip off bias again
        self.layers = layers
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))

        # Part 1: Unsupervised pretraining
        for i in xrange(self.nlayers):
            if i > 0:
                logger.info('layer %d: setting inputs to output of previous '
                            'layer' % i)
                # transform all inputs to generate data for next layer
                out_shape = (inputs.shape[0],
                             self.layers[i - 1].s_hid_plus.shape[1] - 1)
                outputs = self.backend.zeros(out_shape)
                for batch in xrange(num_batches):
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, nrecs)
                    self.positive(inputs[start_idx:end_idx], i - 1)
                    prev_end = self.layers[i - 1].s_hid_plus.shape[1] - 1
                    outputs[start_idx:end_idx] = (self.layers[i -
                                                  1].s_hid_plus[:, 0:prev_end])
                inputs = outputs
                logger.info('inputs (%d, %d) weights (%d,%d)' %
                            (inputs.shape[0], inputs.shape[1],
                             self.layers[i].weights.shape[0],
                             self.layers[i].weights.shape[1]))
                # If we are in the penultimate layer, append labels to the
                # visibles ...
            for epoch in xrange(self.num_epochs):
                error = 0.0
                for batch in xrange(num_batches):
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, nrecs)
                    self.positive(inputs[start_idx:end_idx], i)
                    self.negative(inputs[start_idx:end_idx], i)
                    self.update(self.learning_rate, epoch, self.momentum, i)
                    error += self.cost.apply_function(
                        inputs[start_idx:end_idx],
                        self.layers[i].x_minus[:, 0:(self.layers[i].
                                                     x_minus.shape[1] - 1)])
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / num_batches))
        # Part 2: up-down finetuning ... [not implemented yet]

    def lcreate(self, backend, nin, conf, pos):
        activation = Factory.create(type=conf['activation'])
        # Add 1 for the bias input.
        return RBMLayer(conf['name'], backend,
                        self.batch_size, pos,
                        self.learning_rate,
                        nin + 1,
                        nout=conf['num_nodes'] + 1,
                        activation=activation,
                        weight_init=conf['weight_init'])

    def positive(self, inputs, i):
        """Wrapper for RBMLayer.positive"""
        self.layers[i].positive(inputs)
        return None

    def negative(self, inputs, i):
        """Wrapper for RBMLayer.negative"""
        self.layers[i].negative(inputs)
        return None

    def update(self, epsilon, epoch, momentum, i):
        """Wrapper for RBMLayer.update"""
        self.layers[i].update(epsilon, epoch, momentum)
