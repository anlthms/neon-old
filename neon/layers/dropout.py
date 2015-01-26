# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Single neural network layer in which activations are randomly turned off
according to a specific Bernoulli probability threshold.
"""

import logging
from neon.layers.layer import Layer
from neon.util.param import opt_param

logger = logging.getLogger(__name__)


class DropOutLayer(Layer):
    """
    Dropout layer randomly kills activations from being passed on at each
    fprop call.
    Uses parameter 'keep' as the threshhold above which to retain activation.
    During training, the mask is applied, but during inference, we switch
    off the random dropping.
    Make sure to set train mode to False during inference.

    Attributes:
        keep (numeric, optional): The Bernoulli success probability, indicating
                                  the cutoff below which we keep an activation.
                                  Defaults to 0.5, and should lie in range
                                  [0, 1].
    """
    def initialize(self, kwargs):
        opt_param(self, ['keep'], 0.5)
        super(DropOutLayer, self).initialize(kwargs)
        if self.prev_layer.is_local:
            self.is_local = True
            self.nifm = self.nofm = self.prev_layer.nofm
            self.ifmshape = self.ofmshape = self.prev_layer.ofmshape
        self.nout = self.nin
        self.keepmask = self.backend.empty((self.nin, self.batch_size))
        self.train_mode = True
        self.allocate_output_bufs()

    def fprop(self, inputs):
        if (self.train_mode):
            self.backend.fill_uniform_thresh(self.keepmask, self.keep)
            self.backend.multiply(self.keepmask, self.keep, out=self.keepmask)
            self.backend.multiply(inputs, self.keepmask, out=self.output)
        else:
            self.backend.multiply(inputs, self.keep, out=self.output)

    def bprop(self, error):
        if self.deltas is not None:
            self.backend.multiply(error, self.keepmask, out=self.deltas)

    def set_train_mode(self, mode):
        self.train_mode = mode
