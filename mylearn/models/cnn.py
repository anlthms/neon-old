"""
Contains code to train convnet models and run inference.
"""

import logging

from mylearn.models.layer import LayerWithNoBias
from mylearn.models.layer import ConvLayer, MaxPoolingLayer
from mylearn.models.mlp import MLP
from mylearn.util.factory import Factory

logger = logging.getLogger(__name__)


class CNN(MLP):
    """
    Convnet class
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if isinstance(self.cost, str):
            self.cost = Factory.create(type=self.cost)

    def lcreate(self, backend, nin, conf):
        if conf['connectivity'] == 'full':
            activation = Factory.create(type=conf['activation'])
            return LayerWithNoBias(conf['name'], backend, nin,
                                   nout=conf['num_nodes'],
                                   activation=activation,
                                   weight_init=conf['weight_init'])
        if conf['connectivity'] == 'conv':
            input_shape = conf['input_shape'].split()
            ifmshape = (int(input_shape[0]), int(input_shape[1]))
            filter_shape = conf['filter_shape'].split()
            fshape = (int(filter_shape[0]), int(filter_shape[1]))
            return ConvLayer(conf['name'], backend,
                             batch_size=self.batch_size,
                             nifm=conf['num_input_channels'],
                             ifmshape=ifmshape,
                             fshape=fshape,
                             nfilt=conf['num_filters'],
                             weight_init=conf['weight_init'])
        if conf['connectivity'] == 'mpool':
            input_shape = conf['input_shape'].split()
            ifmshape = (int(input_shape[0]), int(input_shape[1]))
            pooling_shape = conf['pooling_shape'].split()
            pshape = (int(pooling_shape[0]), int(pooling_shape[1]))
            return MaxPoolingLayer(conf['name'], backend,
                                   batch_size=self.batch_size,
                                   nfm=conf['num_channels'],
                                   ifmshape=ifmshape,
                                   pshape=pshape,
                                   weight_init=conf['weight_init'])
