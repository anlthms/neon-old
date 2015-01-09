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

        self.link_and_initialize(self.layers, kwargs)
        for lp in [self.classlayers, self.stylelayers]:
            lp[-1].set_previous_layer(lp[-2])
            lp[-1].initialize(kwargs)

        assert self.layers[-1].nout <= 2 ** 15

    def fprop(self):
        super(Balance, self).fprop()
        for ll in [self.classlayers[-1], self.stylelayers[-1]]:
            ll.fprop(ll.prev_layer.output)

    def bprop(self):
        for path, skip_act in zip(self.pathways, [False, True, False]):
            self.class_layer.skip_act = skip_act
            for ll, nl in zip(reversed(path), reversed(path[1:] + [None])):
                error = None if nl is None else nl.berror
                ll.bprop(error)

    def get_reconstruction_output(self):
        return self.out_layer.output

    def generate_output(self, inputs):
        y = inputs
        for layer in self.layers[1:]:
            layer.fprop(y)
            y = layer.output
            if layer is self.branch_layer:
                y[self.zidx:] = self.backend.wrap(self.zparam)
