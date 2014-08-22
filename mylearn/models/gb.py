"""
Contains code to train Google Brain models and run inference.
"""

import logging
import math

from mylearn.models.layer import LayerWithNoBias
from mylearn.models.layer import LocalFilteringLayer
from mylearn.models.layer import L2PoolingLayer, LCNLayer
from mylearn.models.mlp import MLP
from mylearn.util.factory import Factory

logger = logging.getLogger(__name__)


class GB(MLP):
    """
    Google Brain class
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if isinstance(self.cost, str):
            self.cost = Factory.create(type=self.cost)
        if isinstance(self.pretraining_cost, str):
            self.pretraining_cost = Factory.create(type=self.pretraining_cost)

    def pretrain(self, datasets):
        logger.info('commencing unsupervised pretraining')
        inputs = datasets[0].get_inputs(train=True)['train']
        nrecs, nin = inputs.shape
        self.backend = datasets[0].backend
        self.backend.rng_init()
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        self.trainable_layers = []
        layers = []
        self.temp = []
        for i in xrange(self.nlayers):
            layer = self.lcreate(self.backend, nin, self.layers[i], i)
            if isinstance(layer, LocalFilteringLayer):
                self.trainable_layers.append(i)
                self.temp.append(self.backend.zeros((self.batch_size, nin)))
            logger.info('created layer:\n\t%s' % str(layer))
            layers.append(layer)
            nin = layer.nout
        self.layers = layers

        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for ind in range(len(self.trainable_layers)):
            layer = self.layers[self.trainable_layers[ind]]
            for epoch in xrange(self.num_pretraining_epochs):
                error = 0.0
                for batch in xrange(num_batches):
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, nrecs)
                    output = inputs[start_idx:end_idx]
                    # Forward propagate the input all the way to
                    # the layer that we are pretraining. 
                    for i in range(self.trainable_layers[ind]):
                        self.layers[i].fprop(output)
                        output = self.layers[i].output

                    layer.pretrain(output)
                    error += self.pretraining_cost.apply_function(
                        self.backend, layer.recon,
                        output, self.temp[ind])
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / num_batches))

    def train(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing supervised training')
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
        nrecs, nin = inputs.shape
        self.backend.rng_init()
        tempbuf = self.backend.zeros((self.batch_size, targets.shape[1]))
        self.temp = [tempbuf, tempbuf.copy()]

        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.fprop(inputs[start_idx:end_idx])
                self.bprop(targets[start_idx:end_idx],
                           inputs[start_idx:end_idx],
                           epoch, self.momentum)
                error += self.cost.apply_function(self.backend,
                                                  self.layers[-1].output,
                                                  targets[start_idx:end_idx],
                                                  self.temp)
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))

    def fit(self, datasets):
        self.pretrain(datasets)
        self.train(datasets)

    def lcreate(self, backend, nin, conf, pos):
        if conf['connectivity'] == 'full':
            activation = Factory.create(type=conf['activation'])
            return LayerWithNoBias(conf['name'], backend,
                                   self.batch_size, pos,
                                   self.learning_rate,
                                   nin,
                                   nout=conf['num_nodes'],
                                   activation=activation,
                                   weight_init=conf['weight_init'])
        if conf['connectivity'] == 'lf':
            input_shape = conf['input_shape'].split()
            ifmshape = (int(input_shape[0]), int(input_shape[1]))
            filter_shape = conf['filter_shape'].split()
            fshape = (int(filter_shape[0]), int(filter_shape[1]))
            return LocalFilteringLayer(conf['name'], backend,
                                       self.batch_size, pos,
                                       self.learning_rate,
                                       nifm=conf['num_input_channels'],
                                       ifmshape=ifmshape,
                                       fshape=fshape,
                                       stride=conf['stride'],
                                       weight_init=conf['weight_init'])
        if conf['connectivity'] == 'l2pool':
            input_shape = conf['input_shape'].split()
            ifmshape = (int(input_shape[0]), int(input_shape[1]))
            pooling_shape = conf['pooling_shape'].split()
            pshape = (int(pooling_shape[0]), int(pooling_shape[1]))
            return L2PoolingLayer(conf['name'], backend,
                                  self.batch_size, pos,
                                  nfm=conf['num_channels'],
                                  ifmshape=ifmshape,
                                  pshape=pshape,
                                  stride=conf['stride'])
        if conf['connectivity'] == 'lcn':
            input_shape = conf['input_shape'].split()
            ifmshape = (int(input_shape[0]), int(input_shape[1]))
            filter_shape = conf['filter_shape'].split()
            fshape = (int(filter_shape[0]), int(filter_shape[1]))
            return LCNLayer(conf['name'], backend,
                            self.batch_size, pos,
                            nifm=conf['num_channels'],
                            ifmshape=ifmshape,
                            fshape=fshape,
                            stride=conf['stride'])
