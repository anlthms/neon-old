# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Neural network layers involving the application of convolutional filters.
"""

import logging
from neon.backends.cpu import CPU
from neon.layers.layer import WeightLayer
from neon.util.param import opt_param

logger = logging.getLogger(__name__)


class ConvLayer(WeightLayer):

    """
    Convolutional layer.
    """

    def __init__(self, **kwargs):
        self.is_local = True
        super(ConvLayer, self).__init__(**kwargs)
        opt_param(self, ['local_conv'], False)

    def initialize(self, kwargs):
        super(ConvLayer, self).initialize(kwargs)
        self.initialize_local()
        if self.pad != 0 and isinstance(self.backend, CPU):
            raise NotImplementedError('pad != 0, for CPU backend in ConvLayer')

        self.bias_shape = (self.nout, 1)

        self.allocate_output_bufs()
        self.allocate_param_bufs()
        opt_param(self, ['prodbuf', 'bpropbuf', 'updatebuf'], None)
        if isinstance(self.backend, CPU):
            self.prodbuf = self.backend.empty((self.nofm, self.batch_size))
            self.bpropbuf = self.backend.empty((self.fsize, self.batch_size))
            self.updatebuf = self.backend.empty(self.weights.shape)

    def set_weight_shape(self):
        if hasattr(self, 'local_conv') and self.local_conv:
            weight_shape = (self.fsize * self.ofmsize, self.nofm)
        else:
            weight_shape = (self.fsize, self.nofm)
        opt_param(self, ['weight_shape'], weight_shape)

    def fprop(self, inputs):
        self.backend.fprop_conv(out=self.pre_act, inputs=inputs,
                                weights=self.weights, ofmshape=self.ofmshape,
                                ofmsize=self.ofmsize,
                                ofmlocs=self.ofmlocs, ifmshape=self.ifmshape,
                                links=self.links, nifm=self.nifm,
                                padding=self.pad, stride=self.stride,
                                ngroups=1, fpropbuf=self.prodbuf,
                                local=self.local_conv)
        if self.use_biases is True:
            self.backend.add(self.pre_act, self.biases, out=self.pre_act)
        self.activation.fprop_func(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        inputs = self.prev_layer.output
        self.activation.bprop_func(self.backend, self.pre_act, error)
        if self.deltas is not None:
            self.backend.bprop_conv(out=self.deltas, weights=self.weights,
                                    deltas=error, ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=self.ofmlocs,
                                    ifmshape=self.ifmshape, links=self.links,
                                    padding=self.pad, stride=self.stride,
                                    nifm=self.nifm, ngroups=1,
                                    bpropbuf=self.bpropbuf,
                                    local=self.local_conv)

        upm = self.utemp if self.accumulate else self.updates

        self.backend.update_conv(out=upm[0], inputs=inputs,
                                 weights=self.weights, deltas=error,
                                 ofmshape=self.ofmshape,
                                 ofmsize=self.ofmsize,
                                 ofmlocs=self.ofmlocs,
                                 ifmshape=self.ifmshape, links=self.links,
                                 nifm=self.nifm, padding=self.pad,
                                 stride=self.stride, ngroups=1,
                                 fwidth=self.fshape[-1],
                                 updatebuf=self.updatebuf,
                                 local=self.local_conv,
                                 layer=self)

        if self.use_biases is True:
            self.backend.sum(error, axes=1, out=upm[1])
        if self.accumulate:
            self.backend.add(upm[0], self.updates[0], out=self.updates[0])
            if self.use_biases is True:
                self.backend.add(upm[1], self.updates[1], out=self.updates[1])
