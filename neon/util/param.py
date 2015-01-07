# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains various functions for checking and setting required and optional
parameters.
"""

def req_param(obj, paramlist):
    for param in paramlist:
        if not hasattr(obj, param):
            raise ValueError("req param %s missing for %s" % (param, obj.name))


def opt_param(obj, paramlist, default_value=None):
    for param in paramlist:
        if not hasattr(obj, param):
            setattr(obj, param, default_value)
