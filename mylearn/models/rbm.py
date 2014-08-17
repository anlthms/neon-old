"""
Simple restricted Boltzmann Machine model.
"""

import logging
import math
from mylearn.models.layer import RBMLayer
from mylearn.models.model import Model
from mylearn.util.factory import Factory

logger = logging.getLogger(__name__)


class RBM(Model):

    """
    Restricted Boltzmann Machine with binary visible and binary hidden units
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
            layer = self.lcreate(self.backend, nin, self.layers[i])
            logger.info('created layer:\n\t%s' % str(layer))
            layers.append(layer)
        self.layers = layers

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.positive(inputs.take(range(start_idx, end_idx), axis=0))
                self.negative(inputs.take(range(start_idx, end_idx), axis=0))
                self.update(self.learning_rate, epoch, self.momentum)

                error += self.cost.apply_function(
                    inputs.take(range(start_idx, end_idx), axis=0),
                    self.layers[0].x_minus.take(range(self.layers[0].x_minus.shape[1] - 1), axis=1))
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))

        import matplotlib.pyplot as plt
        for i in range(100):
            plt.subplot(10, 10, i + 1)
            plt.imshow(self.layers[0].weights.take(i, axis=0).take(
                range(784), axis=1).raw().reshape(28, 28))
            plt.gray()
        plt.show()

    def lcreate(self, backend, nin, conf):
        activation = Factory.create(type=conf['activation'])
        # Add 1 for the bias input.
        return RBMLayer(conf['name'], backend, nin + 1,
                        nout=conf['num_nodes'] + 1,
                        activation=activation,
                        weight_init=conf['weight_init'])

    def positive(self, inputs):
        """Wrapper for RBMLayer.positive"""
        # self.layers[0].positive_explicit_bias(inputs)
        self.layers[0].positive(inputs)
        return None

    def negative(self, inputs):
        """Wrapper for RBMLayer.negative"""
        # no error here since it's unspervised.
        # self.layers[0].negative_explicit_bias(inputs)
        self.layers[0].negative(inputs)
        return None

    def update(self, epsilon, epoch, momentum):
        """Wrapper for RBMLayer.update"""
        # self.layers[0].update_explicit_bias(epsilon, epoch, momentum)
        self.layers[0].update(epsilon, epoch, momentum)
