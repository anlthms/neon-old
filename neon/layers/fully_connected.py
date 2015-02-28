# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
from neon.layers.layer import WeightLayer
from neon.transforms.batch_norm import BatchNorm
from neon.util.param import req_param, opt_param

logger = logging.getLogger(__name__)


class FCLayer(WeightLayer):
    """
    Fully connected feed-forward neural network layer.

    Attributes:
        nin (integer): number of input connections (from previous layer).
        nout (integer): number of output activations.
    """
    def initialize(self, kwargs):
        super(FCLayer, self).initialize(kwargs)
        req_param(self, ['nin', 'nout'])
        self.bias_shape = (self.nout, 1)

        self.allocate_output_bufs()

        if self.batch_norm:
            self.bn = BatchNorm()
            kwargs['layer'] = self
            self.bn.initialize(kwargs)

        self.allocate_param_bufs()

    def set_weight_shape(self):
        opt_param(self, ['weight_shape'], (self.nout, self.nin))

    def fprop(self, inputs):
        self.backend.fprop_fc(out=self.pre_act, inputs=inputs,
                              weights=self.weights, layer=self)
        if self.use_biases is True:
            self.backend.add(self.pre_act, self.biases, out=self.pre_act)

        if self.batch_norm:
            self.bn.fprop_func(self.backend, self.pre_act, self.pre_act)

        self.activation.fprop_func(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        inputs = self.prev_layer.output
        self.activation.bprop_func(self.backend, self.pre_act, error,
                                   self.skip_act)

        upm = self.utemp if self.accumulate else self.updates

        if self.batch_norm:
            self.bn.bprop_func(self.backend, self.pre_act, error,
                               self.skip_act)
            upm = upm[2:]

        if self.deltas is not None:
            self.backend.bprop_fc(out=self.deltas, weights=self.weights,
                                  deltas=error, layer=self)

        self.backend.update_fc(out=upm[0], inputs=inputs,
                               deltas=error, layer=self)
        if self.use_biases is True:
            self.backend.sum(error, axes=1, out=upm[1])

        if self.accumulate:
            self.backend.add(upm[0], self.updates[0], out=self.updates[0])
            if self.use_biases is True:
                self.backend.add(upm[1], self.updates[1], out=self.updates[1])
