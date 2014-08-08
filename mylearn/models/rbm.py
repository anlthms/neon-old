"""
Simple restricted Boltzmann Machine model.
"""

import logging
import math
from ipdb import set_trace as trace

from mylearn.models.layer import RBMLayer # (u) created RBMLayer...
from mylearn.models.model import Model


logger = logging.getLogger(__name__)


class RBM(Model):
    """
    Restricted Boltzmann Machine with binary visible and binary hidden units
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) # (u) here the 'model' part of the yaml comes in

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing model fitting')
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
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
            # (u) we die on str(layer), maybe lcreate below is not ok?
            layer = self.lcreate(self.backend, nin, self.layers[i])
            ## logger.info('created layer:\n\t%s' % str(layer)) # (u) calls Layer.__str__, the layer passed in does not have the correct self.weights?
            layers.append(layer)
            nin = layer.nout
        self.layers = layers

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.fprop(inputs.take(range(start_idx, end_idx), axis=0))
                self.bprop(targets.take(range(start_idx, end_idx), axis=0))
                self.update(inputs.take(range(start_idx, end_idx), axis=0),
                            self.learning_rate, epoch, self.momentum)
                error += self.loss_fn(self.layers[-1].output,
                                      targets.take(range(start_idx, end_idx),
                                                   axis=0))
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))
            # for layer in self.layers:
            #    logger.info('layer:\n\t%s' % str(layer))
        


    def lcreate(self, backend, nin, conf):
        if conf['connectivity'] == 'full':
            # Add 1 for the bias input.
            return RBMLayer(conf['name'], backend, nin + 1,
                         nout=conf['num_nodes']+1,  # (u) bias for both layers
                         act_fn=conf['activation_fn'],
                         weight_init=conf['weight_init'])

    def positive(self, inputs):
        layers[0].positive(y)
        y = layers[0].output
        return y

    def negative(self, targets):
        error = self.layers[0].error()
        self.layers[0].negative(error)

    def update(self, inputs, epsilon, epoch, momentum):
        self.layers[0].update(inputs, epsilon, epoch, momentum)
        for i in xrange(1, self.nlayers):
            self.layers[i].update(self.layers[i - 1].output, epsilon, epoch,
                                  momentum)
