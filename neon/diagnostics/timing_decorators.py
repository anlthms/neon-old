# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for measuring FLOPS on backend mop calls.
"""

import numpy as np
import pycuda.driver as drv  # for timing
import traceback  # for tracing back where the function was called from
from functools import wraps

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

multis = {'fprop_fc' : 2,
          'bprop_fc' : 2,
          'update_fc': 2
         }




def record_flops(mult, shape_list, func_name):
    """
    decorator idea needs some work, function calls are very different
    """
    # mult, shape_list, func_name are visible below!
    def record_flops_decorator(func):
        # funct is visible below!
        @wraps(func)
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
        # return a function that can be called with: args, kwargs
        # this means, func_wrap = record_flops_decorator(fprop_fc)
        #     stuff = func_wrap(self, *args, **kwargs)
        print "RETURNING FUNC_WRAPPER, needs to be a method!!!!!", func_wrapper
        return func_wrapper
    # return a function that can be called with: func
    # this means,  record_flops_dec = record_flops(2, ..., bprop_fc)
    #      stuff = record_flops_dec(func)
    print "returning record_flops_decorator", record_flops_decorator
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

