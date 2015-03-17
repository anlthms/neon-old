# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls. Functions are decorated

We are looking for 1s per iteration, breaks down to 24op with 40ms each.
We see 20ms ops, must account for EW better.

EW 0.4ms overall does not seem right, maybe the bundling does not work? Yes
greater is 1.3 already!


Manual sanity check for FC layer: 2*4000*4000*128 = 4GF, here takes 1ms for 4TF.
In soumith, it's not timed, doh!
conv2 fprop: 57GF (check: 2 times 5*5*64 x 128  times 5*5*64 x 1*1*192 gives 1*1*192 x 128 -> 80M in 27*27=730 locations totals 60GF ok!)
which takes here 16ms for 3.5TF.
In Soumith, this layer ccn2 fprop takes 252ms. His numbers are bananas!
"""

import logging

import numpy as np
import traceback  # for tracing back where the function was called from
from functools import wraps
from time import time  as now # for timing.
from collections import defaultdict

logger = logging.getLogger(__name__)


class Decorators(object):
    # things that are mulitplied together to compute number of operations
    shapes = {'fprop_fc' : [('inputs', 0), ('inputs', 1), ('weights', 0)],
              'bprop_fc' : [('deltas', 0), ('deltas', 1), ('weights', 1)],
              'update_fc': [('inputs', 0), ('inputs', 1), ('deltas', 0)],
              # for convolution, FLOPs are
              'fprop_conv' : [('inputs', 0), ('inputs', 1), ('weights', 0)],
              'bprop_conv' : [('deltas', 0), ('deltas', 1), ('weights', 1)],
              'update_conv': [('inputs', 0), ('inputs', 1), ('deltas', 0)]}

    # constant factors in front of FLOP terms
    multipliers = {'fprop_fc' : 2, 'bprop_fc' : 2, 'update_fc': 2,
                   'fprop_conv' : 2, 'bprop_conv' : 2, 'update_conv': 2}

    # elementwise operations: multipliers and arg position
    ew_mult_pos = {'logistic': 4, 'rectlin': 1, 'sum': 1, 'mean': 1, 'var': 1,
                   'sqrt':1, 'add': 1, 'subtract': 1, 'multiply': 1,
                   'divide': 1, 'greater': 1, 'not_equal': 1,
                   'clip': 2, 'log': 1, 'argmax': 10, 'softmax': 10,
                   'gdm_compound': 5, 'gdmwd_compound': 10
                  } # 'zeros': 1, 'ones': 1, 'empty': 1, 'array': 1, 'copy_from': 1,
                    # 'fprop_pool': 1, 'bprop_pool': 1,

    def __init__(self, **kwargs):
        """
        Initialize output dictionaries where the timing diagnostics are stored.
        Jerry-rigging the backend using the monkey trick
        """
        kwargs['backend'].time_dict = defaultdict(list)
        kwargs['backend'].flop_dict = defaultdict(list)
        kwargs['backend'].paren_dic = defaultdict(list)
        kwargs['backend'].layer_dic = defaultdict(list)

    def decorate(self, backend, function_list):
        """
        Replaces the @decorators in the backend function. Go through the list
        of functions to be decorated and wrap them with the correct parameters
        """
        for call in function_list['decorate_fc']:
            print "wrapping", call, "with", self.multipliers[call],
            print "and", self.shapes[call]
            orig_func = getattr(backend, call)
            wrapped_func = self.record_flops_fc(orig_func)
            setattr(backend, call, wrapped_func)
        for call in function_list['decorate_conv']:
            print "wrapping", call, "with", self.multipliers[call],
            print "and", self.shapes[call]
            orig_func = getattr(backend, call)
            wrapped_func = self.record_flops_conv(orig_func)
            setattr(backend, call, wrapped_func)
        for call in function_list['decorate_ew']:
            print "wrapping", call, "ew"
            orig_func = getattr(backend, call)
            wrapped_func = self.record_flops_ew(orig_func)
            setattr(backend, call, wrapped_func)

    def record_flops_fc(self, func):
        """
        This function takes a list of tensors and shape indices, and multiplies
        the shapes together. This works well for dot products. The flops are scaled
        with a global multiplier taken from the 'multipliers' dict.
        """
        func_name = func.__name__
        if func_name not in self.multipliers:
            raise ValueError("Cannot record flops for function: %s" % func_name)
        #@wraps
        def func_wrapper(*arguments, **kwargs):
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            if 'weights' in kwargs:
                layer_name = kwargs['weights'].name
            else:
                layer_name = 'undefined'
            #logger.info("MOP call: %s from parent %s", func_name, parent_func_name)
            #####################
            tic = self.start_me()
            #####################
            retval = func(*arguments, **kwargs)
            #####################
            msecs = self.stop_me(tic)
            #####################
            flop = self.multipliers[func_name]
            for (matrix,dim) in self.shapes[func_name]:
                flop *= kwargs[matrix].shape[dim]
            func.__self__.time_dict[func_name].append(msecs / 1000.)
            func.__self__.flop_dict[func_name].append(flop)
            func.__self__.paren_dic[func_name].append(parent_func_name)
            func.__self__.layer_dic[func_name].append(layer_name)
            return retval
        return func_wrapper

    def record_flops_conv(self, func):
        """
        Custom function for fprop_conv, fix up later.
        """
        func_name = func.__name__
        if func_name not in self.multipliers:
            raise ValueError("Cannot record flops for function: %s" % func_name)
        #@wraps
        def func_wrapper(*arguments, **kwargs):
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            layer_name = kwargs['weights'].name
            #import pdb; pdb.set_trace()

            #logger.info("MOP call: %s from parent %s", func_name, parent_func_name)
            tic = self.start_me()
            #####################
            retval = func(*arguments, **kwargs)
            #####################
            msecs = self.stop_me(tic)
            if func_name == 'fprop_conv':
                '''
                fprop: convolution between input and filter.
                on out, inputs, weights, ofmshape, ofmsize

                comes out wrong, impossibly fast. Need to use the CPU way:
                '''
                # N = kwargs['inputs'].shape[1] # 128
                # C = kwargs['nifm'] # 3
                # PQ = kwargs['ofmshape'][0] # 28
                # RS = np.sqrt(kwargs['weights'].shape[0]/C) # 5
                # K = kwargs['weights'].shape[1] # 16
                # mads = RS**3 * PQ**2 * C * K * N
                # adds = K**2 * K * (C-1)
                mads = kwargs['ofmsize'] \
                         * kwargs['weights'].shape[0] \
                         * kwargs['weights'].shape[1] \
                         * kwargs['inputs'].shape[1]  # ofmsize784 weights(75,16) inputs[1]128
                adds = 0
                flop = 2 * mads + adds
                #print "fprop flopps", flop
            elif func_name == 'bprop_conv':
                '''
                bprop: convolution between zero padded delta and kernel
                loop(ofmsize) w0 w1 d1 (taked from CPU backend)
                '''
                mads = kwargs['ofmsize'] \
                         * kwargs['weights'].shape[0] \
                         * kwargs['weights'].shape[1] \
                         * kwargs['deltas'].shape[1]
                # adds = kwargs['ofmsize'] \  # kind of a bizarre object
                #          * kwargs['bpropbuf'].shape[0] \
                #          * kwargs['bpropbuf'].shape[1] # looking for (400, 128)
                adds = kwargs['ofmsize'] * kwargs['out'].shape[1] * kwargs['weights'].shape[0]
                flop = 2 * mads + adds
                #print "bprop flopps", flop
            elif func_name == 'update_conv':
                '''
                update: convolution between input data and delta matrix
                taken from CPU backend
                '''
                #import pdb; pdb.set_trace()
                mads = kwargs['ofmsize'] \
                         * kwargs['deltas'].shape[1] \
                         * kwargs['out'].shape[1] \
                         * kwargs['out'].shape[0] # ous both  400, 32
                adds = kwargs['ofmsize'] \
                         * kwargs['out'].shape[0] \
                         * kwargs['out'].shape[1] #  out, updatebuf is obj
                flop = 2 * mads + adds
                #print "update flopps", flop
            func.__self__.time_dict[func_name].append(msecs / 1000.)
            func.__self__.flop_dict[func_name].append(flop)
            func.__self__.paren_dic[func_name].append(parent_func_name)
            func.__self__.layer_dic[func_name].append(layer_name)
            return retval
        return func_wrapper

    def record_flops_ew(self, func):
        """
        This function wraps elementwise operations, where the first or second
        argument is a tensor. FLOPS are computed by multiplying the two
        dimensions. The scalar multiplier is taken from 'ew_mult_pos'
        TODO: remove dimension from 'ew_mult_pos', it's inferred automatically.
        """
        func_name = func.__name__
        if func_name not in self.ew_mult_pos:
            raise ValueError("Cannot record flops for ew function: %s" % func_name)
        #@wraps
        def func_wrapper(*arguments, **kwargs):
            """
            Note args have a live of their own (reseved keyword) and shall not be
            used. kwargs on the other hand are just a dict.
            """
            if 'weights' in kwargs:
                layer_name = kwargs['weights'].name
            else:
                layer_name = 'anon'
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            tic = self.start_me()
            #####################
            retval = func(*arguments, **kwargs)
            #####################
            msecs = self.stop_me(tic)
            array_arg = 1 if (type(arguments[0]) is float) else 0

            flop = (self.ew_mult_pos[func_name]
                    * arguments[array_arg].shape[0]
                    * arguments[array_arg].shape[1])
            func.__self__.time_dict[func_name].append(msecs / 1000.)
            func.__self__.flop_dict[func_name].append(flop)
            func.__self__.paren_dic[func_name].append(parent_func_name)
            func.__self__.layer_dic[func_name].append(layer_name)
            return retval
        return func_wrapper


class MaxDecorators(Decorators):
    """
    These are max-backend specific decorators that use pycuda for timing.
    """

    def __init__(self, **kwargs):
        '''init used to hide import until we have a backend'''
        import pycuda.driver as drv  # for timing.

        self.start  = drv.Event()
        self.end    = drv.Event()

        super(MaxDecorators, self).__init__(**kwargs)

    def start_me(self):
        #
        tic = self.start.record()
        return tic

    def stop_me(self, tic):
        self.end.record()
        self.end.synchronize()
        msecs = self.end.time_since(self.start)
        return msecs


class CudanetDecorators(Decorators):
    """
    decorators for cudanet (TODO: Update for new format)
    """

    def __init__(self, **kwargs):
        '''init used to hide import until we have a backend'''
        import cudanet # pass
        self.sync = cudanet.sync_stream

        super(CudanetDecorators, self).__init__(**kwargs)

    def start_me(self):
        tic = now()
        return tic

    def stop_me(self, tic):
        self.sync() # syncstream is a GPU backend function.
        msecs = 1000. * (now() - tic)
        return msecs
