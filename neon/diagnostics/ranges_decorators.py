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
        for call in function_list['decorate_ranges']:
            print "wrapping", call
            orig_func = getattr(self.backend, call)
            print "seinding", orig_func
            wrapped_func = self.print_ranges(orig_func)
            setattr(self.backend, call, wrapped_func)

    def print_ranges(self, func):
        """
        This function takes a list of tensors and shape indices, and multiplies
        the shapes together. This works well for dot products. The flops are
        scaled with a global multiplier taken from the 'multipliers' dict.
        """
        func_name = func.__name__
        if func_name not in self.supported_funcs:
            raise ValueError("Cannot compute ranges for : %s" % func_name)

        @wraps(func)
        def func_wrapper(*arguments, **kwargs):
            parent_func_name = traceback.extract_stack(limit=2)[-2][2]
            be = self.backend
            # orig. function call
            retval = func(*arguments, **kwargs)

            # new plotting stuff
            print "\nbackend call to", func_name, "from", parent_func_name
            for item in ['weights', 'deltas', 'out', 'inputs']:
                if item in kwargs:
                    print item,
                    the_min = be.zeros((1, 1))
                    the_max = be.zeros((1, 1))
                    be.min(kwargs[item], axes=None, out=the_min)
                    be.max(kwargs[item], axes=None, out=the_max)
                    print "\tstd", kwargs[item][0:3].asnumpyarray().astype(
                                                        np.float32).std(1),
                    print "\traw", kwargs[item][0, 0:3].asnumpyarray(),
                    print "\tmin", the_min.asnumpyarray(),
                    print "\tmax", the_max.asnumpyarray()

            return retval
        return func_wrapper
