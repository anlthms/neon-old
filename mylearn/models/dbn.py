"""
Simple deep believe net.
"""

import logging
import math
from ipdb import set_trace as trace

from mylearn.models.layer import RBMLayer # (u) created RBMLayer...
from mylearn.models.model import Model
from mylearn.util.factory import Factory

logger = logging.getLogger(__name__)


class DBN(Model):
    """
    deep believe net
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
            nin = layer.nout - 1 # strip off bias again 
        self.layers = layers

 
        for i in xrange(self.nlayers): # learn the layers
            if i>0:
                print 'layer %d: setting inputs to output of previous layer' %i
                self.positive(inputs, i-1) # transform all inputs to generate data for next layer
                inputs = self.layers[i-1].s_hid_plus.take(range(self.layers[i-1].s_hid_plus.shape[1] - 1), axis=1) # this is only one batch, need to compute it for all! 
                logger.info('inputs (%d, %d) weights (%d,%d)' %
                    (inputs.shape[0],inputs.shape[1], self.layers[i].weights.shape[0],self.layers[i].weights.shape[1]) )
            num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
            for epoch in xrange(self.num_epochs):
                error = 0.0
                for batch in xrange(num_batches):
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, nrecs)
                    self.positive(inputs.take(range(start_idx, end_idx), axis=0),i) # dies! 
                    self.negative(inputs.take(range(start_idx, end_idx), axis=0),i)
                    self.update(self.learning_rate, epoch, self.momentum,i)
                    # this error measures how much one step of CD1 moves away from the data.
                    #trace()
                    error += self.cost.apply_function( inputs.take(range(start_idx, end_idx), axis=0), #(100, 784)
                                                       self.layers[i].x_minus.take(range(self.layers[i].x_minus.shape[1] - 1), axis=1) )# for implicit bias 
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / num_batches))
 

    
    def lcreate(self, backend, nin, conf):
        # instantiate the activation function class from string name given
        activation = Factory.create(type=conf['activation'])
        # Add 1 for the bias input.
        return RBMLayer(conf['name'], backend, nin + 1,
                     nout=conf['num_nodes']+1,  # (u) bias for both layers?
                     activation=activation,
                     weight_init=conf['weight_init'])

    def positive(self, inputs,i):
        """Wrapper for RBMLayer.positive"""
        #self.layers[0].positive_explicit_bias(inputs) # (u) pass in data y and compute positive pass
        self.layers[i].positive(inputs) # (u) pass in data y and compute positive pass
        y = self.layers[i].output # (u) output is set by running positive. 
        return y

    def negative(self, inputs,i):
        """Wrapper for RBMLayer.negative"""
        # no error here since it's unspervised. 
        #self.layers[0].negative_explicit_bias(inputs)
        self.layers[i].negative(inputs)
        y = self.layers[i].output # (u) output is set by running positive. 
        return y

    def update(self, epsilon, epoch, momentum,i):
        """Wrapper for RBMLayer.update"""
        #self.layers[i].update_explicit_bias(epsilon, epoch, momentum) 
        self.layers[i].update(epsilon, epoch, momentum) 
