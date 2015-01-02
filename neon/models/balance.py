# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train a balance network, containing both supervised and
unsupervised branches and multiple cost functions.
Requires model to specify prev layers at each layer to build the layer graph
"""

import logging
from neon.models.layer import BranchLayer
from neon.models.mlp import MLPB
from neon.transforms.cross_entropy import CrossEntropy

logger = logging.getLogger(__name__)


class Balance(MLPB):

    def __init__(self, **kwargs):
        self.dist_mode = None
        self.__dict__.update(kwargs)
        for req_param in ['layers', 'batch_size']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)
        self.result = 0
        kwargs = {"backend": self.backend, "batch_size": self.batch_size,
                  "accumulate": True}

        self.data_layer = self.layers[0]
        self.cost_layer = self.classlayers[-1]
        self.out_layer = self.layers[-2]
        self.class_layer = self.classlayers[-2]
        self.branch_layer = self.stylelayers[-2]
        self.pathways = [self.layers, self.classlayers, self.stylelayers]

        self.data_layer.initialize(kwargs)

        for (pathway, startidx) in zip(self.pathways, [1, -2, -1]):
            pl = pathway[startidx-1]
            for layer in pathway[startidx:]:
                layer.set_previous_layer(pl)
                layer.initialize(kwargs)
                pl = layer

        assert self.layers[-1].nout <= 2 ** 15

    def fprop(self):
        for (pathway, startidx) in zip(self.pathways, [0, -2, -1]):
            y = pathway[startidx-1].output if startidx != 0 else None
            for layer in pathway[startidx:]:
                layer.fprop(y)
                y = layer.output

    def bprop(self):
        for pathway in self.pathways:
            error = None
            for layer in reversed(pathway):
                layer.bprop(error)
                error = layer.berror

    def get_reconstruction_output(self):
        return self.out_layer.output

    def generate_output(self, inputs):
        y = inputs
        for layer in self.layers[1:]:
            layer.fprop(y)
            y = layer.output
            if layer is self.branch_layer:
                y[self.zidx:] = self.backend.wrap(self.zparam)
