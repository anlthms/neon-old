# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains neural network layers that are composed of multiple sublayer parts.
"""

import logging
from neon.layers.layer import Layer
from neon.util.compat import range
from neon.util.param import req_param

logger = logging.getLogger(__name__)


class CompositeLayer(Layer):
    """
    Generic parent for Branch and List layer that deals with sublayer list

    Attributes:
        sublayers (list of Layer): Individual layers that make up this
                                   compositional layer.
    """
    def initialize(self, kwargs):
        super(CompositeLayer, self).initialize(kwargs)
        req_param(self, ['sublayers'])
        for l in self.sublayers:
            self.backend.begin()
            l.initialize(kwargs)
            self.backend.end()

    def update(self, epoch):
        for l in self.sublayers:
            self.backend.begin()
            l.update(epoch)
            self.backend.end()

    def set_train_mode(self, mode):
        for sublayer in self.sublayers:
            self.backend.begin()
            sublayer.set_train_mode(mode)
            self.backend.end()


class BranchLayer(CompositeLayer):
    """
    Branch layer is composed of a list of other layers concatenated with one
    another.

    During fprop, it concatenates the component outputs and passes it on.
    During bprop, it splits the backward errors into the components and
    accumulates into common deltas
    """

    def __init__(self, **kwargs):
        super(BranchLayer, self).__init__(**kwargs)
        self.nout = reduce(lambda x, y: x + y.nout, self.sublayers, 0)

    def set_previous_layer(self, pl):
        super(BranchLayer, self).set_previous_layer(pl)
        for l in self.sublayers:
            self.backend.begin()
            l.set_previous_layer(pl)
            self.backend.end()

    def initialize(self, kwargs):
        super(BranchLayer, self).initialize(kwargs)

        self.startidx = [0] * len(self.sublayers)
        self.endidx = [0] * len(self.sublayers)
        self.endidx[0] = self.sublayers[0].nout
        for i in range(1, len(self.sublayers)):
            self.backend.begin()
            self.endidx[i] = self.endidx[i - 1] + self.sublayers[i].nout
            self.startidx[i] = self.endidx[i - 1]
            self.backend.end()

        self.allocate_output_bufs()

    def fprop(self, inputs):
        for (s_l, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            self.backend.begin()
            s_l.fprop(inputs)
            self.output[si:ei] = s_l.output
            self.backend.end()

    def bprop(self, error):
        for (s_l, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            self.backend.begin()
            s_l.bprop(error[si:ei])
            self.backend.end()

        if self.deltas is not None:
            self.deltas.fill(0.0)
            for subl in self.sublayers:
                self.backend.begin()
                self.backend.add(self.deltas, subl.deltas, out=self.deltas)
                self.backend.end()


class ListLayer(Layer):
    """
    List layer is composed of a list of other layers stacked on top of one
    another.

    During fprop, it simply fprops along the chain.
    During bprop, it splits the backward errors into the components and
    accumulates into common deltas
    """
    def set_previous_layer(self, pl):
        super(ListLayer, self).set_previous_layer(pl)
        for l in self.sublayers:
            self.backend.begin()
            l.set_previous_layer(pl)
            pl = l
            self.backend.end()

    def initialize(self, kwargs):
        super(ListLayer, self).initialize(kwargs)
        self.output = self.sublayers[-1].output
        self.deltas = self.sublayers[0].deltas
        self.nout = self.sublayers[-1].nout
        if self.sublayers[-1].is_local is True:
            self.nofm = self.sublayers[-1].nofm
            self.ofmshape = self.sublayers[-1].ofmshape

    def fprop(self, inputs):
        y = inputs
        for l in self.sublayers:
            self.backend.begin()
            l.fprop(y)
            y = l.output
            self.backend.end()

    def bprop(self, error):
        error = None
        for l in reversed(self.sublayers):
            self.backend.begin()
            l.bprop(error)
            self.backend.end()
