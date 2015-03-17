# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Decorators for dumping parameter ranges and raw values. Useful for tracking
overflow and underflow when using limited precision formats.
"""

import logging

import numpy as np
import traceback  # for tracing back where the function was called from
from functools import wraps

logger = logging.getLogger(__name__)


class Decorators(object):

    supported_funcs = ['fprop_fc', 'bprop_fc', 'update_fc',
                       'fprop_conv', 'bprop_conv', 'update_conv']

    def __init__(self, backend):
        self.backend = backend

    def decorate(self, function_list):
        """
        Replaces the @decorators in the backend function. Go through the list
        of functions to be decorated and wrap them with the correct parameters
        """
        for call in function_list['decorate']:
            print "wrapping", call
            orig_func = getattr(self.backend, call)
            print "seinding", orig_func
            wrapped_func = self.print_ranges(orig_func)
            setattr(self.backend, call, wrapped_func)


    def print_ranges(self, func):
        """
        This function takes a list of tensors and shape indices, and multiplies
        the shapes together. This works well for dot products. The flops are scaled
        with a global multiplier taken from the 'multipliers' dict.
        """
        func_name = func.__name__
        if func_name not in self.supported_funcs and func_name is not 'func_wrapper':
            raise ValueError("Cannot compute ranges for function: %s" % func_name)
        @wraps(func)
        def func_wrapper(*arguments, **kwargs):
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]

            # orig. function call
            retval = func(*arguments, **kwargs)

            # new plotting stuff
            print "\nbackend call to",func_name , "from", parent_func_name
            for item in ['weights', 'deltas', 'out', 'inputs']:
                if item in kwargs:
                    print item,
                    print "\tstd", kwargs[item].asnumpyarray().astype(np.float32).std(1)[0:3],
                    print "\traw", kwargs[item][0,0:3].asnumpyarray(),
                    print "\tmin", kwargs[item].asnumpyarray().min(),
                    print "\tmax", kwargs[item].asnumpyarray().max()

            return retval
        return func_wrapper


