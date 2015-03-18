# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls
"""

import logging

import traceback  # for tracing back where the function was called from
from functools import wraps
from time import time as now
from collections import defaultdict

logger = logging.getLogger(__name__)


class Decorators(object):
    # things that are mulitplied together to compute number of operations
    shapes = {'fprop_fc':  [('inputs', 0), ('inputs', 1), ('weights', 0)],
              'bprop_fc':  [('deltas', 0), ('deltas', 1), ('weights', 1)],
              'update_fc': [('inputs', 0), ('inputs', 1), ('deltas', 0)],
              # for convolution, FLOPs are
              'fprop_conv':  [('inputs', 0), ('inputs', 1), ('weights', 0)],
              'bprop_conv':  [('deltas', 0), ('deltas', 1), ('weights', 1)],
              'update_conv': [('inputs', 0), ('inputs', 1), ('deltas', 0)]}

    # constant factors in front of FLOP terms
    multipliers = {'fprop_fc':   2, 'bprop_fc':   2, 'update_fc':   2,
                   'fprop_conv': 2, 'bprop_conv': 2, 'update_conv': 2}

    # elementwise operations: multipliers and arg position
    ew_mult_pos = {'logistic': 4, 'rectlin': 1, 'sum': 1, 'mean': 1, 'var': 1,
                   'sqrt': 1, 'add': 1, 'subtract': 1, 'multiply': 1,
                   'divide': 1, 'greater': 1, 'not_equal': 1,
                   'clip': 2, 'log': 1, 'argmax': 10, 'softmax': 10,
                   'gdm_compound': 5, 'gdmwd_compound': 10}

    def __init__(self, backend):
        """
        Initialize output dictionaries where the timing diagnostics are stored.
        Jerry-rigging the backend using the monkey trick
        """
        backend.time_dict = defaultdict(list)
        backend.flop_dict = defaultdict(list)
        backend.paren_dic = defaultdict(list)
        backend.layer_dic = defaultdict(list)
        self.backend = backend

    def decorate(self, function_list):
        """
        Replaces the @decorators in the backend function. Go through the list
        of functions to be decorated and wrap them with the correct parameters
        """
        for call in function_list['decorate_fc']:
            print "wrapping", call, "with", self.multipliers[call],
            print "and", self.shapes[call]
            orig_func = getattr(self.backend, call)
            wrapped_func = self.record_flops_fc(orig_func)
            setattr(self.backend, call, wrapped_func)
        for call in function_list['decorate_conv']:
            print "wrapping", call, "with", self.multipliers[call],
            print "and", self.shapes[call]
            orig_func = getattr(self.backend, call)
            wrapped_func = self.record_flops_conv(orig_func)
            setattr(self.backend, call, wrapped_func)
        for call in function_list['decorate_ew']:
            print "wrapping", call, "ew"
            orig_func = getattr(self.backend, call)
            wrapped_func = self.record_flops_ew(orig_func)
            setattr(self.backend, call, wrapped_func)

    def record_flops_fc(self, func):
        """
        This function takes a list of tensors and shape indices, and multiplies
        the shapes together. This works well for dot products. The flops are
        scaled with a global multiplier taken from the 'multipliers' dict.
        """
        func_name = func.__name__
        if func_name not in self.multipliers:
            raise ValueError("Cannot record flops for: %s" % func_name)

        @wraps(func)
        def func_wrapper(*arguments, **kwargs):
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            if 'weights' in kwargs:
                layer_name = kwargs['weights'].name
            else:
                layer_name = 'undefined'
            #####################
            tic = self.start_me()
            #####################
            retval = func(*arguments, **kwargs)
            #####################
            msecs = self.stop_me(tic)
            #####################
            flop = self.multipliers[func_name]
            for (matrix, dim) in self.shapes[func_name]:
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
            raise ValueError("Cannot record flops for: %s" % func_name)

        @wraps(func)
        def func_wrapper(*arguments, **kwargs):
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            layer_name = kwargs['weights'].name
            tic = self.start_me()
            #####################
            retval = func(*arguments, **kwargs)
            #####################
            msecs = self.stop_me(tic)
            if func_name == 'fprop_conv':
                '''
                fprop: convolution between input and filter.
                '''
                mads = (kwargs['ofmsize'] *
                        kwargs['weights'].shape[0] *
                        kwargs['weights'].shape[1] *
                        kwargs['inputs'].shape[1])
                adds = 0
                flop = 2 * mads + adds
            elif func_name == 'bprop_conv':
                '''
                bprop: convolution between zero padded delta and kernel
                loop(ofmsize) w0 w1 d1 (taked from CPU backend)
                '''
                mads = (kwargs['ofmsize'] *
                        kwargs['weights'].shape[0] *
                        kwargs['weights'].shape[1] *
                        kwargs['deltas'].shape[1])
                adds = (kwargs['ofmsize'] *
                        kwargs['out'].shape[1] *
                        kwargs['weights'].shape[0])
                flop = 2 * mads + adds
            elif func_name == 'update_conv':
                '''
                update: convolution between input data and delta matrix
                taken from CPU backend
                '''
                mads = (kwargs['ofmsize'] *
                        kwargs['deltas'].shape[1] *
                        kwargs['out'].shape[1] *
                        kwargs['out'].shape[0])
                adds = (kwargs['ofmsize'] *
                        kwargs['out'].shape[0] *
                        kwargs['out'].shape[1])
                flop = 2 * mads + adds
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
        """
        func_name = func.__name__
        if func_name not in self.ew_mult_pos:
            raise ValueError("Cannot record flops for: %s" % func_name)

        @wraps(func)
        def func_wrapper(*arguments, **kwargs):
            """
            Note 'args' have a live of their own (reseved keyword) and shall
            not be used. kwargs on the other hand are just a dict.
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

            flop = (self.ew_mult_pos[func_name] *
                    arguments[array_arg].shape[0] *
                    arguments[array_arg].shape[1])
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

        self.start = drv.Event()
        self.end = drv.Event()

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
        import cudanet
        self.sync = cudanet.sync_stream

        super(CudanetDecorators, self).__init__(**kwargs)

    def start_me(self):
        tic = now()
        return tic

    def stop_me(self, tic):
        self.sync()  # syncstream is a GPU backend function.
        msecs = 1000. * (now() - tic)
        return msecs
