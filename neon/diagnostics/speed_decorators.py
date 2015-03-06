# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls.
"""

import numpy as np
import pycuda.driver as drv  # for timing
import traceback  # for tracing back where the function was called from

start  = drv.Event()
end    = drv.Event()


# things that are mulitplied together to compute number of operations
shapes = {'fprop_fc' : [('inputs', 0), ('inputs', 1), ('weights', 0)],
          'bprop_fc' : [('deltas', 0), ('deltas', 1), ('weights', 1)],
          'update_fc': [('inputs', 0), ('inputs', 1), ('deltas', 0)],  # deltas correct??
          'logistic' : [('x', 0), ('x', 1)],
          'rectlin'  : [('x', 0), ('x', 1)],
          'rectlin_derivative': [('x', 0), ('x', 1)],
          'sum'      : [('tsr', 0), ('tsr', 1)]

         }

def record_flops(mult, shape_list, func_name):
    """
    decorator idea needs some work, function calls are very different
    """
    def record_flops_decorator(func):
        def func_wrapper(self, *args, **kwargs):
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            #print("MOP call: " + func_name + " from parent " + parent_func_name)
            start.record()
            #####################
            func(self, *args, **kwargs)
            #####################
            end.record()
            end.synchronize()
            msecs = end.time_since(start)

            flop = mult
            for (matrix,dim) in shape_list:
                flop *= kwargs[matrix].shape[dim]
            self.time_dict[func_name].append(msecs / 1000.)
            self.flop_dict[func_name].append(flop)
            self.paren_dic[func_name].append(parent_func_name)
        return func_wrapper
    return record_flops_decorator

def record_flops_ew(mult, arg_pos, func_name):
    """
    for ew functions called with args, not kwargs.
    """
    def record_flops_decorator(func):
        def func_wrapper(self, *args, **kwargs):
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            #print("EW call: " + func_name + " from parent " + parent_func_name)
            start.record()
            #####################
            func(self, *args, **kwargs)
            #####################
            end.record()
            end.synchronize()
            msecs = end.time_since(start)

            flop = mult
            #print "args", args
            flop *= args[arg_pos].shape[0]
            flop *= args[arg_pos].shape[1]
            self.time_dict[func_name].append(msecs / 1000.)
            self.flop_dict[func_name].append(flop)
            self.paren_dic[func_name].append(parent_func_name)
        return func_wrapper
    return record_flops_decorator

