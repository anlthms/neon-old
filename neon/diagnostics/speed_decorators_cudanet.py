# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls.
"""

import numpy as np
#from neon.backends.gpu import GPU
from time import time



# things that are mulitplied together to compute number of operations
shapes = {'fprop_fc' : [('inputs', 0), ('inputs', 1), ('weights', 0)],
          'bprop_fc' : [('deltas', 0), ('deltas', 1), ('weights', 1)],
          'update_fc': [('inputs', 0), ('inputs', 1), ('deltas', 0)],  # deltas correct??
          'logistic' : [('x', 0), ('x', 1)],
          'rectlin'  : [('x', 0), ('x', 1)],
          'rectlin_derivative': [('x', 0), ('x', 1)],
          'sum'      : [('tsr', 0), ('tsr', 1)]

         }


# decorators for cudanet

def record_flops(mult, shape_list, func_name):
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

def record_flops_ew(mult, arg_pos, func_name):
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
