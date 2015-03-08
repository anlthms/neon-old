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
          'update_fc': [('inputs', 0), ('inputs', 1), ('deltas', 0)]}

# constant factors in front of FLOP terms
multipliers = {'fprop_fc' : 2, 'bprop_fc' : 2, 'update_fc': 2}

# elementwise operations: multipliers and arg position
ew_mult_pos = { 'logistic': (4,0), 'rectlin': (1,0),
                 'add':(1,0), 'subtract':(1,1), 'multiply': (1,0),
                 'divide': (1,0), 'greater': (1,0), 'not_equal':(1,0),
                 'clip': (2,0), 'log': (1,0), 'argmax': (1,0)
              }

def scott_record_flops(func):
    func_name = func.__name__
    if func_name not in multipliers:
        raise ValueError("Cannot record flops for function: %s" % func_name)
    #@wraps
    def func_wrapper(*arguments, **kwargs):
        parent_func_name = traceback.extract_stack(limit=2)[-2][2]
        #print("MOP call: " + func_name + " from parent " + parent_func_name)
        start.record()
        #####################
        func(*arguments, **kwargs)
        #####################
        end.record()
        end.synchronize()
        msecs = end.time_since(start)
        #import pdb; pdb.set_trace()
        flop = multipliers[func_name]
        for (matrix,dim) in shapes[func_name]:
            flop *= kwargs[matrix].shape[dim]
        func.__self__.time_dict[func_name].append(msecs / 1000.)
        func.__self__.flop_dict[func_name].append(flop)
        func.__self__.paren_dic[func_name].append(parent_func_name)
    return func_wrapper

def scott_record_flops_ew(func):
    func_name = func.__name__
    if func_name not in ew_mult_pos:
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
        func(*arguments, **kwargs)
        #####################
        end.record()
        end.synchronize()
        msecs = end.time_since(start)
        #import pdb; pdb.set_trace()
        array_arg = 1 if (type(arguments[0]) is float) else 0
        #flop = ew_mult_pos[func_name][0]
        #flop *= arguments[ew_mult_pos[func_name][1]].shape[0]
        #flop *= arguments[ew_mult_pos[func_name][1]].shape[1]
        flop = (ew_mult_pos[func_name][0] * arguments[array_arg].shape[0]
                                          * arguments[array_arg].shape[1])
        func.__self__.time_dict[func_name].append(msecs / 1000.)
        func.__self__.flop_dict[func_name].append(flop)
        func.__self__.paren_dic[func_name].append(parent_func_name)
    return func_wrapper

