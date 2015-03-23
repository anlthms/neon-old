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
        for subl in self.sublayers:
            subl.initialize(kwargs)

    def __str__(self):
        ret = '{} {}: {} nodes'.format(self.__class__.__name__,
                                       self.name, self.nout)
        ret += ':\n'
        for subl in self.sublayers:
            ret += '\t' + str(subl) + '\n'
        return ret

    def __str__(self):
        ret = '{} {}: {} nodes'.format(self.__class__.__name__,
                                       self.name, self.nout)
        ret += ':\n'
        for l in self.sublayers:
            ret += '\t' + str(l) + '\n'
        return ret

    def update(self, epoch):
        for subl in self.sublayers:
            subl.update(epoch)

    def set_train_mode(self, mode):
        for subl in self.sublayers:
            subl.set_train_mode(mode)


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

    def set_previous_layer(self, pl):
        super(BranchLayer, self).set_previous_layer(pl)
        for subl in self.sublayers:
            subl.set_previous_layer(pl)
        self.nout = reduce(lambda x, y: x + y.nout, self.sublayers, 0)
        if pl is not None:
            self.nin = pl.nout

    def initialize(self, kwargs):
        super(BranchLayer, self).initialize(kwargs)

        self.startidx = [0] * len(self.sublayers)
        self.endidx = [0] * len(self.sublayers)
        self.endidx[0] = self.sublayers[0].nout
        for i in range(1, len(self.sublayers)):
            self.endidx[i] = self.endidx[i - 1] + self.sublayers[i].nout
            self.startidx[i] = self.endidx[i - 1]

        self.allocate_output_bufs()

    def init_dataset(self, dataset):
        for layer in self.sublayers:
            layer.init_dataset(dataset)

    def use_set(self, setname, predict=False):
        for layer in self.sublayers:
            layer.use_set(setname, predict)
        self.num_batches = self.sublayers[0].num_batches

    def reset_counter(self):
        for layer in self.sublayers:
            layer.reset_counter()

    def has_more_data(self):
        return self.sublayers[0].has_more_data()

    def has_set(self, setname):
        return self.sublayers[0].has_set(setname)

    def cleanup(self):
        for layer in self.sublayers:
            layer.cleanup()

    def fprop(self, inputs):
        for (s_l, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            s_l.fprop(inputs)
            self.output[si:ei] = s_l.output
        self.targets = self.sublayers[0].targets

    def set_deltas_buf(self, delta_pool, offset):
        if self.prev_layer is None:
            return
        if self.prev_layer.is_data:
            return

        self.deltas = self.backend.zeros(self.delta_shape, self.deltas_dtype)
        for subl in self.sublayers:
            subl.set_deltas_buf(delta_pool, offset)

    def fprop(self, inputs):
        for (subl, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            subl.fprop(inputs)
            self.output[si:ei] = subl.output
        self.targets = self.sublayers[0].targets

    def bprop(self, error):
        if self.deltas is not None:
            self.deltas.fill(0.0)
        for (subl, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            subl.bprop(error[si:ei])
            if self.deltas is not None:
                self.backend.add(self.deltas, subl.deltas, out=self.deltas)


class ListLayer(CompositeLayer):
    """
    List layer is composed of a list of other layers stacked on top of one
    another.

    During fprop and bprop, it simply operates along the chain.
    """
    def set_previous_layer(self, pl):
        super(ListLayer, self).set_previous_layer(pl)
        for subl in self.sublayers:
            subl.set_previous_layer(pl)
            pl = subl
        self.nout = self.sublayers[-1].nout

    def initialize(self, kwargs):
        super(ListLayer, self).initialize(kwargs)
        self.output = self.sublayers[-1].output
        self.deltas = self.sublayers[0].deltas
        if self.sublayers[-1].is_local is True:
            self.nofm = self.sublayers[-1].nofm
            self.ofmshape = self.sublayers[-1].ofmshape

    def init_dataset(self, dataset):
        self.sublayers[0].init_dataset(dataset)

    def use_set(self, setname, predict=False):
        self.sublayers[0].use_set(setname, predict)
        self.num_batches = self.sublayers[0].num_batches

    def reset_counter(self):
        self.sublayers[0].reset_counter()

    def has_more_data(self):
        return self.sublayers[0].has_more_data()

    def has_set(self, setname):
        return self.sublayers[0].has_set(setname)

    def cleanup(self):
        self.sublayers[0].cleanup()

    def fprop(self, inputs):
        y = inputs
        for l in self.sublayers:
            l.fprop(y)
            y = l.output
        self.targets = self.sublayers[0].targets

    def bprop(self, error):
        for l in reversed(self.sublayers):
            l.bprop(error)
            error = l.deltas
=======
    def set_deltas_buf(self, delta_pool, offset):
        self.ninmax = max(map(lambda x: x.nin, self.sublayers))
        delta_shape = (2 * self.ninmax, self.batch_size)
        assert len(self.sublayers) > 1
        delta_type = self.sublayers[1].deltas_dtype
        self.delta_pool = self.backend.zeros(delta_shape, delta_type)
        for idx, subl in enumerate(self.sublayers):
            offset = (idx % 2) * self.ninmax
            subl.set_deltas_buf(self.delta_pool, offset)

        if self.prev_layer is None:
            return
        if self.prev_layer.is_data:
            return
        self.deltas = self.sublayers[0].deltas

    def fprop(self, inputs):
        for subl in self.sublayers:
            subl.fprop(inputs)
            inputs = subl.output

    def bprop(self, error):
        for subl in reversed(self.sublayers):
            subl.bprop(error)
            error = subl.deltas
>>>>>>> cf716c88b86600e17abc1960201b2f65da08b97e
