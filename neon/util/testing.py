# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Utility functions which help with running tests.
"""

import numpy as np

from neon.backends.backend import Tensor


def assert_tensor_equal(actual, desired):
    """
    Ensures that Tensor array contents are identical in shape and each element.

    Arguments:
        actual (object): The first Tensor for comparison.
        desired (object): The expected value to be compared against.

    Raises:
        AssertionError: if any of the elements or shapes differ.
    """
    assert_tensor_near_equal(actual, desired, tolerance=0)


def assert_tensor_near_equal(actual, desired, tolerance=1e-7):
    """
    Ensures that Tensor array contents are equal (up to the specified
    tolerance).

    Arguments:
        actual (object): The first value for comparison.
        desired (object): The expected value to be compared against.
        tolerance (float, optional): Threshold tolerance.  Items are considered
                                     equal if their absolute difference does
                                     not exceed this value.

    Raises:
        AssertionError: if the objects differ.
    """
    if isinstance(desired, Tensor):
        desired = desired.raw()
    if isinstance(actual, Tensor):
        actual = actual.raw()
    np.testing.assert_allclose(actual, desired, atol=tolerance, rtol=0)
