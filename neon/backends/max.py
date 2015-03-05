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
        Original dot
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

    @st.record_flops_ew(mult=4, arg_pos=0, func_name='ew')
    def logistic(self, x, out):
        self.nl.sig(x, out=out)
        # self.multiply(x, -1.0, out=out)
        # self.exp(out, out=out)
        # self.add(out, 1.0, out=out)
        # self.reciprocal(out, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew')
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

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew')
    def add(self, left, right, out):
        """assignment"""
        self.nl.add(left, right, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=1, func_name='ew')
    def subtract(self, left, right, out):
        """assignment"""
        self.nl.subtract(left, right, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew')
    def multiply(self, left, right, out):
        """assignment"""
        self.nl.multiply(left, right, out=out)
        return out

    #@st.record_flops_ew(mult=1, arg_pos=0, func_name='reduce')
    def divide(self, left, right, out):
        """assignment"""
        self.nl.divide(left, right, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew')
    def greater(self, left, right, out):
        """assignment"""
        self.nl.greater(left, right, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew')
    def not_equal(self, left, right, out):
        """assignment"""
        self.nl.not_equal(left, right, out=out)
        return out

    @st.record_flops_ew(mult=2, arg_pos=0, func_name='ew')
    def clip(self, a, a_min, a_max, out):
        """assignment"""
        self.nl.clip(a, a_min, a_max, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='ew')
    def log(self, a, out):
        """assignment"""
        self.nl.log(a, out=out)
        return out

    @st.record_flops_ew(mult=1, arg_pos=0, func_name='reduce')
    def argmax(self, a, out, axis=1):
        """assignment"""
        self.nl.argmax(a, out=out, axis=axis)
        return out
