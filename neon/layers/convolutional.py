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
import numpy as np
np.set_printoptions(linewidth=200)
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

        opt_param(self, ['shared_bias'], True)
        if self.shared_bias:
            self.bias_shape = (self.nofm, 1)
            self.bias_expand = self.backend.empty((self.nout, 1))
        else:
            self.bias_shape = (self.nout, 1)

        self.allocate_output_bufs()
        self.allocate_param_bufs()

        opt_param(self, ['prodbuf', 'bpropbuf', 'updatebuf'], None)
        if isinstance(self.backend, CPU):
            self.prodbuf = self.backend.empty((self.nofm, self.batch_size))
            self.bpropbuf = self.backend.empty((self.fsize, self.batch_size))
            self.updatebuf = self.backend.empty(self.weights.shape)

        if hasattr(self.backend, 'nl'):
            self.conv_params = self.backend.nl.conv_layer(
                N=self.batch_size, C=self.nifm, K=self.nofm,
                D=1, H=self.ifmshape[0], W=self.ifmshape[1], T=1,
                R=self.fshape[0], S=self.fshape[1],
                pad_d=0, pad_h=self.pad, pad_w=self.pad,
                str_d=1, str_h=self.stride, str_w=self.stride)
            self.prodbuf = self.bpropbuf = self.updatebuf = self.conv_params


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
                                padding=self.negpad, stride=self.stride,
                                ngroups=1, fpropbuf=self.prodbuf,
                                local=self.local_conv)
        if 'conv-' in self.name:
            print "\nbackend call to fprop_conv", self.name
            print "std weights", self.weights.asnumpyarray().astype(np.float32).std(1)[0:4], "\traw",  self.weights[0,0:4].asnumpyarray() # var
            print "std inputs", inputs.asnumpyarray().astype(np.float32).std(1)[0:4], "\traw",  inputs[0,0:4].asnumpyarray()
            print "std pre_act", self.pre_act.asnumpyarray().astype(np.float32).std(1)[0:4], "\traw",  self.pre_act[0,0:4].asnumpyarray()
        if self.use_biases is True:
            if self.shared_bias:
                self.pre_act = self.pre_act.reshape(
                    (self.nofm, self.ofmsize * self.batch_size))
                self.backend.add(self.pre_act, self.biases, out=self.pre_act)
                self.pre_act = self.pre_act.reshape(
                    (self.nofm * self.ofmsize, self.batch_size))
            else:
                self.backend.add(self.pre_act, self.biases, out=self.pre_act)


        if self.batch_norm:
            self.bn.fprop_func(self.backend, self.pre_act, self.pre_act)

        self.activation.fprop_func(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        inputs = self.prev_layer.output
        self.activation.bprop_func(self.backend, self.pre_act, error,
                                   self.skip_act)

        upm = self.utemp if self.accumulate else self.updates
        u_idx = 0
        if self.batch_norm:
            self.bn.bprop_func(self.backend, self.pre_act, error,
                               self.skip_act)
            u_idx = 2

        if self.deltas is not None:
            #self.backend.multiply(error, float(1000), out=error)
            self.backend.bprop_conv(out=self.deltas, weights=self.weights,
                                    deltas=error, ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=self.ofmlocs,
                                    ifmshape=self.ifmshape, links=self.links,
                                    padding=self.negpad, stride=self.stride,
                                    nifm=self.nifm, ngroups=1,
                                    bpropbuf=self.bpropbuf,
                                    local=self.local_conv)
            #self.backend.divide(error, float(1000), out=error) # scale back
            if 'conv-' in self.name:
                print "\nbackend call to BPROP_CONV", self.name
                print "weights\tstd", self.weights.asnumpyarray().astype(np.float32).std(1)[0:4], "\traw",  self.weights[0,0:4].asnumpyarray()# var over feature maps
                print "error \tstd", error.asnumpyarray().astype(np.float32).std(1)[0:4], "\traw",  error[0,0:4].asnumpyarray() # var over batchsize
                print "deltas\tstd", self.deltas.asnumpyarray().astype(np.float32).std(1)[0:4], "\traw",  self.deltas[0,0:4].asnumpyarray() # BUG! Deltas are nonzer0, std is zero!
            #import pdb; pdb.set_trace()
        self.backend.update_conv(out=upm[u_idx], inputs=inputs,
                                 weights=self.weights, deltas=error,
                                 ofmshape=self.ofmshape,
                                 ofmsize=self.ofmsize,
                                 ofmlocs=self.ofmlocs,
                                 ifmshape=self.ifmshape, links=self.links,
                                 nifm=self.nifm, padding=self.negpad,
                                 stride=self.stride, ngroups=1,
                                 fwidth=self.fshape[-1],
                                 updatebuf=self.updatebuf,
                                 local=self.local_conv,
                                 layer=self)

        if self.use_biases is True:
            # We can't reshape the error buffer since it might be global buffer
            if self.shared_bias:
                self.backend.sum(error, axes=1, out=self.bias_expand)
                self.bias_expand = self.bias_expand.reshape(
                    (self.nofm, self.ofmsize))
                self.backend.sum(self.bias_expand, axes=1, out=upm[u_idx+1])
                self.bias_expand = self.bias_expand.reshape(
                    (self.nofm * self.ofmsize, 1))
            else:
                self.backend.sum(error, axes=1, out=upm[u_idx+1])

        if self.accumulate:
            self.backend.add(upm[u_idx], self.updates[u_idx],
                             out=self.updates[u_idx])
            if self.use_biases is True:
                self.backend.add(upm[u_idx+1], self.updates[u_idx+1],
                                 out=self.updates[u_idx+1])
