"""
Simple restricted Boltzmann Machine model.
"""

import logging
import math
from ipdb import set_trace as trace

from mylearn.models.layer import RBMLayer # (u) created RBMLayer...
from mylearn.models.model import Model
from mylearn.util.factory import Factory

logger = logging.getLogger(__name__)


class RBM(Model):
    """
    Restricted Boltzmann Machine with binary visible and binary hidden units
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) # (u) here the 'model' part of the yaml comes in
        if isinstance(self.cost, str):
            self.cost = Factory.create(type=self.cost)

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing model fitting')
        inputs = datasets[0].get_inputs(train=True)['train']
        #targets = datasets[0].get_targets(train=True)['train']
        nrecs, nin = inputs.shape
        self.backend = datasets[0].backend
        self.backend.rng_init()
        #self.loss_fn = getattr(self.backend, self.loss_fn) # (u) gets backend.loss_fn, does not exist? This turns the string 'cross_entropy' (form self.__dict__) into the method Numpy.cross_entropy 
        #self.de = self.backend.get_derivative(self.loss_fn)
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        layers = []
        for i in xrange(self.nlayers):
            layer = self.lcreate(self.backend, nin, self.layers[i])
            logger.info('created layer:\n\t%s' % str(layer)) # (u) calls Layer.__str__, the layer passed in does not have the correct self.weights?
            layers.append(layer)
        self.layers = layers

        #trace() # (u) now the meat: generate the correct layers and train them 

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
                # this error measures how much one step of CD1 moves away from the data.
                #trace()
                error += self.cost.apply_function( inputs.take(range(start_idx, end_idx), axis=0), #(100, 784)
                                                   self.layers[0].x_minus.take(range(self.layers[0].x_minus.shape[1] - 1), axis=1) )# for implicit bias 
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))
            # for layer in self.layers:
            #    logger.info('layer:\n\t%s' % str(layer))
        #trace()
        import matplotlib.pyplot as plt
        for i in range(100):
            plt.subplot(10,10,i+1)
            plt.imshow(self.layers[0].weights.take(i, axis=0).take(range(784), axis=1).raw().reshape(28,28))
            plt.gray()
        plt.show()
        #trace()
        # filters look reasonable, but this code should really not be here. 
        


    def lcreate(self, backend, nin, conf):
        # instantiate the activation function class from string name given
        activation = Factory.create(type=conf['activation'])
        # Add 1 for the bias input.
        return RBMLayer(conf['name'], backend, nin + 1,
                     nout=conf['num_nodes']+1,  # (u) bias for both layers?
                     activation=activation,
                     weight_init=conf['weight_init'])

    def positive(self, inputs):
        """Wrapper for RBMLayer.positive"""
        #self.layers[0].positive_explicit_bias(inputs) # (u) pass in data y and compute positive pass
        self.layers[0].positive(inputs) # (u) pass in data y and compute positive pass
        y = self.layers[0].output # (u) output is set by running positive. 
        return y

    def negative(self, inputs):
        """Wrapper for RBMLayer.negative"""
        # no error here since it's unspervised. 
        #self.layers[0].negative_explicit_bias(inputs)
        self.layers[0].negative(inputs)
        y = self.layers[0].output # (u) output is set by running positive. 
        return y

    def update(self, epsilon, epoch, momentum):
        """Wrapper for RBMLayer.update"""
        #self.layers[0].update_explicit_bias(epsilon, epoch, momentum) 
        self.layers[0].update(epsilon, epoch, momentum) 
