# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls. Functions are decorated
in a
"""

import numpy as np
import pycuda.driver as drv  # for timing
import traceback  # for tracing back where the function was called from
from functools import wraps

start  = drv.Event()
end    = drv.Event()


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
    ew_mult_pos = { 'logistic': 4, 'rectlin': 1,
                     'add':1, 'subtract': 1, 'multiply': 1, 'divide': 1,
                     'greater': 1, 'not_equal':1, 'clip': 2, 'log': 1, 'argmax': 1
                  }

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


class MaxDecorators(Decorators):

    """
    These are max-backend specific decorators that use pycuda for timing.
    """

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
            #print("MOP call: " + func_name + " from parent " + parent_func_name)
            start.record()
            #####################
            retval = func(*arguments, **kwargs)
            #####################
            end.record()
            end.synchronize()
            msecs = end.time_since(start)
            #import pdb; pdb.set_trace()
            flop = self.multipliers[func_name]
            for (matrix,dim) in self.shapes[func_name]:
                flop *= kwargs[matrix].shape[dim]
            func.__self__.time_dict[func_name].append(msecs / 1000.)
            func.__self__.flop_dict[func_name].append(flop)
            func.__self__.paren_dic[func_name].append(parent_func_name)
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
            #print("MOP call: " + func_name + " from parent " + parent_func_name)
            start.record()
            #####################
            retval = func(*arguments, **kwargs)
            #####################
            end.record()
            end.synchronize()
            msecs = end.time_since(start)
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
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            start.record()
            #####################
            retval = func(*arguments, **kwargs)
            #####################
            end.record()
            end.synchronize()
            msecs = end.time_since(start)
            array_arg = 1 if (type(arguments[0]) is float) else 0

            flop = (self.ew_mult_pos[func_name]
                    * arguments[array_arg].shape[0]
                    * arguments[array_arg].shape[1])
            func.__self__.time_dict[func_name].append(msecs / 1000.)
            func.__self__.flop_dict[func_name].append(flop)
            func.__self__.paren_dic[func_name].append(parent_func_name)
            return retval
        return func_wrapper


class CudanetDecorators(Decorators):
    """
    decorators for cudanet (TODO: Update for new format)
    """

    def record_flops(self, mult, shape_list, func_name):
        """
        decorator idea needs some work, function calls are very different
        """
        def record_flops_decorator(func):
            def func_wrapper(self, *args, **kwargs):
                tic = time()
                #####################
                func(self, *args, **kwargs)
                #####################
                self.sync_stream()
                msecs = 1000. * (time() - tic)

                flop = mult
                for (matrix,dim) in shape_list:
                    flop *= kwargs[matrix].shape[dim]
                self.time_dict[func_name].append(msecs / 1000.)
                self.flop_dict[func_name].append(flop)
            return func_wrapper
        return record_flops_decorator

    def record_flops_ew(self, mult, arg_pos, func_name):
        """
        for ew functions called with args, not kwargs.
        """
        def record_flops_decorator(func):
            def func_wrapper(self, *args, **kwargs):
                tic = time()
                #####################
                func(self, *args, **kwargs)
                #####################
                self.sync_stream()
                msecs = 1000. * (time() - tic)

                flop = mult
                #print "args", args
                flop *= args[arg_pos].shape[0]
                flop *= args[arg_pos].shape[1]
                self.time_dict[func_name].append(msecs / 1000.)
                self.flop_dict[func_name].append(flop)
            return func_wrapper
        return record_flops_decorator
