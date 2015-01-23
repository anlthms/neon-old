# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train a balance network, containing both supervised and
unsupervised branches and multiple cost functions.
Requires model to specify prev layers at each layer to build the layer graph
"""

import logging
from neon.models.mlp import MLPB
from neon.util.param import req_param

logger = logging.getLogger(__name__)


class Balance(MLPB):

    def __init__(self, **kwargs):
        self.accumulate = True
        super(Balance, self).__init__(**kwargs)
        req_param(self, ['classlayers', 'stylelayers'])
        self.cost_layer = self.classlayers[-1]
        self.out_layer = self.layers[-2]
        self.class_layer = self.classlayers[-2]
        self.branch_layer = self.stylelayers[-2]
        self.pathways = [self.layers, self.classlayers, self.stylelayers]

        for lp in [self.classlayers, self.stylelayers]:
            self.backend.begin()
            lp[-1].set_previous_layer(lp[-2])
            lp[-1].initialize(kwargs)
            self.backend.end()

    def fprop(self):
        super(Balance, self).fprop()
        for ll in [self.classlayers[-1], self.stylelayers[-1]]:
            self.backend.begin()
            ll.fprop(ll.prev_layer.output)
            self.backend.end()

    def bprop(self):
        for path, skip_act in zip(self.pathways, [False, True, False]):
            self.backend.begin()
            self.class_layer.skip_act = skip_act
            for ll, nl in zip(reversed(path), reversed(path[1:] + [None])):
                self.backend.begin()
                error = None if nl is None else nl.deltas
                ll.bprop(error)
                self.backend.end()
            self.backend.end()

    def get_reconstruction_output(self):
        return self.out_layer.output

    def generate_output(self, inputs):
        y = inputs
        for layer in self.layers[1:]:
            self.backend.begin()
            layer.fprop(y)
            y = layer.output
            if layer is self.branch_layer:
                y[self.zidx:] = self.zparam
            self.backend.end()
