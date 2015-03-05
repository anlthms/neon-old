# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Backend wrapper for nervana_lib. Most functions are inherited directly from
the NervanaLib class, and FloatArray is taken from there.
"""
import logging

from neon.backends.backend import Backend
import sys
sys.path.append('/home/users/urs/code/flexgpu')
from nervana_lib import NervanaLib, FloatArray
import pycuda.driver as drv
import numpy as np
from time import time
from collections import defaultdict
from neon.diagnostics import speed_test as st

from neon.util.compat import range


logger = logging.getLogger(__name__)



class MAX(Backend):
    """
    the nl functions usually return opTrees. If we supply out=out, the
    reduction takes places and reduces to out. Otherwise we can use a [:]
    assignment to collapse the tree. Note = will just alias the opTree.

    Everything in here is a reduction.
    """
    def __init__(self, rng_seed, stochastic_round=False):
        self.nl = NervanaLib(stochastic_round=stochastic_round)
        logger.info("Initialized NervanaLib with stochastic_round=%s",
                    stochastic_round)
        self.rng_seed = rng_seed
        self.rng_init()

        # output dictionaries where the timing diagnostics are stored
        self.time_dict = defaultdict(list)
        self.flop_dict = defaultdict(list)


    def rng_init(self):
        seed = None
        if 'rng_seed' in self.__dict__:
            seed = self.rng_seed
            logger.info("Seeding random number generator with: %s", str(seed))
        np.random.seed(seed)

    def uniform(self, low=0.0, high=1.0, shape=1, dtype=None, name=None,
                allocator=drv.mem_alloc):
        """
        generate numpy random number and convert to a FloatArray.
        If called with dype=None it will probably explode
        """
        ary = np.random.uniform(low, high, shape)
        return FloatArray(ary.shape, dtype, allocator=allocator, name=name,
                          rounding=self.nl.round_mode).set(ary)

    def normal(self, loc=0.0, scale=1.0, size=1, dtype=None, name=None,
                allocator=drv.mem_alloc):
        """
        Gaussian/Normal random number sample generation
        """
        ary = np.random.normal(loc, scale, size)
        return FloatArray(ary.shape, dtype, allocator=allocator, name=name,
                          rounding=self.nl.round_mode).set(ary)

    @st.record_flops(mult=2, shape_list=st.shapes['fprop_fc'], func_name='fprop_fc')
    def fprop_fc(self, out, inputs, weights, layer=None):
        """
        Dot calls for fully conneted layer fprop, bprop and update.
        Inputs:
            out
            inputs
            weights

        """
        self.nl.dot(weights, inputs, out)

    @st.record_flops(mult=2, shape_list=st.shapes['bprop_fc'], func_name='bprop_fc')
    def bprop_fc(self, out, weights, deltas, layer=None):
        """
        NervanaLib dot call
        """
        self.nl.dot(weights.T, deltas, out)

    @st.record_flops(mult=2, shape_list=st.shapes['update_fc'], func_name='update_fc') # noqa
    def update_fc(self, out, inputs, deltas, layer=None):
        """
        NervanaLib dot call
        """
        self.nl.dot(deltas, inputs.T, out)



    def fprop_conv(self, out, inputs, weights, ofmshape, ofmsize, ofmlocs,
                   ifmshape, links, nifm, padding, stride, ngroups, fpropbuf,
                   local=False):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).  For this
                              backend we expect these values to be square.
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  convolved outputs for a single receptive
                                  field.  Not used for this backend.
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
        """
        import pdb; pdb.set_trace()
        assert ifmshape[-2] == ifmshape[-1]
        # convert parameters to SG convention
        N
        C = nimf
        H,W = ifmshape
        K = nofm
        R,S = 0,0

        # set up a ConvLayer parameter object
        conv = self.nl.conv_layer(self,
            N, C, K,
            D=1, H=1, W=1,
            T=1, R=1, S=1,
            pad_d=0, pad_h=0, pad_w=0,
            str_d=1, str_h=1, str_w=1,
            scl_d=1, scl_h=1, scl_w=1)
        '''
        N: Number of images in mini-batch
        C: Number of input feature maps
        K: Number of output feature maps

        D: Depth  of input image
        H: Height of input image
        W: Width  of input image

        T: Depth  of filter kernel
        R: Height of filter kernel
        S: Width  of filter kernel
        '''
        self.nl.fprop_conv(conv, I, F, O, alpha=1.0, repeat=1)
        # cudanet.convolution(
        #     weights._tensor, inputs._tensor, out._tensor,
        #     ifmshape[-2], ofmshape[-2], ofmshape[-1], padding, stride, nifm,
        #     ngroups)

    def bprop_conv(self, out, weights, deltas, ofmshape, ofmsize, ofmlocs,
                   ifmshape, links, padding, stride, nifm, ngroups, bpropbuf,
                   local=False):
        """
        Backward propagate the error through a convolutional network layer.
        """
        bprop_conv(conv, F, E, grad_I, alpha=1.0, repeat=1)
        # cudanet.deconvolve_errors(
        #     weights._tensor, deltas._tensor,
        #     out._tensor, ifmshape[-2], ifmshape[-1], ofmshape[-2],
        #     padding, stride, nifm, ngroups)

    def update_conv(self, out, inputs, weights, deltas, ofmshape, ofmsize,
                    ofmlocs, ifmshape, links, nifm, padding, stride, ngroups,
                    fwidth, updatebuf, local=False, layer=None):
        """
        Compute the updated gradient for a convolutional network layer.

        """
        self.nl.update_conv(conv, I, E, grad_F, alpha=1.0, repeat=1)
        # cudanet.deconvolve_wts(
        #     deltas._tensor, inputs._tensor, out._tensor,
        #     ifmshape[-2], ofmshape[-2], ofmshape[-1], fwidth,
        #     padding, stride, nifm, ngroups, ofmshape[-2], local)





    @st.record_flops_ew(mult=4, arg_pos=0, func_name='ew_sig')
    def logistic(self, x, out):
        self.nl.sig(x, out=out)
        # self.multiply(x, -1.0, out=out)
        # self.exp(out, out=out)
        # self.add(out, 1.0, out=out)
        # self.reciprocal(out, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew_relu')
    def rectlin(self, x, out):
        # x and out are the same buffer
        self.nl.maximum(x, 0., out=out)
        return out

    # @st.record_flops(mult=1, shape_list=shapes['rectlin_derivative'], func_name='ew')
    # def rectlin_derivative(self, x, out):
    #     print "reclin ew yo!"
    #     self.nl.greater(x, 0., out=out)
    #     return out

    #@st.record_flops_ew(mult=1, func_name='sum')
    #@st.record_flops_ew(mult=1, func_name='sum')
    # sum done this way breaks add below.
    def sum(self, tsr, axes, out):
        """wrapper to make full reduction possible"""

        if axes is None:
            sze = tsr.shape[0]*tsr.shape[1]
            self.nl.sum_axis(tsr.reshape(sze,1), axis=0, out=out)
        else:
            self.nl.sum_axis(tsr, axis=axes, out=out)
        return out

    def zeros(self, shape, dtype=np.float16):
        """
        wrap. Using default float16 is a little white cheat
        """
        return self.nl.zeros(shape, dtype=dtype)

    def empty(self, shape, dtype=np.float16):
        """
        wrap, cheat on dtype
        """
        return self.nl.empty(shape, dtype=dtype)

    def array(self, ary, dtype=np.float16, name=None, allocator=drv.mem_alloc):
        """
        copy and paste
        """
        return FloatArray(ary.shape, dtype, allocator=allocator, name=name,
                          rounding=self.nl.round_mode).set(ary)

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew_add')
    def add(self, left, right, out):
        """assignment"""
        self.nl.add(left, right, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=1, func_name='ew_sub')
    def subtract(self, left, right, out):
        """assignment"""
        self.nl.subtract(left, right, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew_mul')
    def multiply(self, left, right, out):
        """assignment"""
        self.nl.multiply(left, right, out=out)
        return out

    #@st.record_flops_ew(mult=1, arg_pos=0, func_name='reduce')
    def divide(self, left, right, out):
        """assignment"""
        self.nl.divide(left, right, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew_gre')
    def greater(self, left, right, out):
        """assignment"""
        self.nl.greater(left, right, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew_neq')
    def not_equal(self, left, right, out):
        """assignment"""
        self.nl.not_equal(left, right, out=out)
        return out

    @st.record_flops_ew(mult=2, arg_pos=0, func_name='ew_clip')
    def clip(self, a, a_min, a_max, out):
        """assignment"""
        self.nl.clip(a, a_min, a_max, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew_log')
    def log(self, a, out):
        """assignment"""
        self.nl.log(a, out=out)
        return out

    #@st.record_flops_ew(mult=1, arg_pos=0, func_name='reduce')
    def argmax(self, a, out, axis=1):
        """assignment"""
        self.nl.argmax(a, out=out, axis=axis)
        return out
