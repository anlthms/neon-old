#!/usr/bin/env/python

import numpy as np

from neon.backends.flexpoint import (FlexpointTensor, flexpt_dtype)
# from neon.util.testing import assert_tensor_equal


class TestFlexpointTensor(object):

    def __init__(self):
        # this code gets called prior to each test
        self.default_dtype = flexpt_dtype(sign_bit=True, int_bits=4,
                                          frac_bits=11, overflow=0, rounding=0)

    def test_empty_creation(self):
        tns = FlexpointTensor([], self.default_dtype)
        # this behavior deviates from numpy dimension reduction (0,)
        # but is expected given that we need to currently create a 2D structure
        assert tns.shape == (0, 1)

    def test_1d_creation(self):
        tns = FlexpointTensor([1, 2, 3, 4], self.default_dtype)
        assert tns.shape == (4, 1)

    def test_2d_creation(self):
        tns = FlexpointTensor([[1, 2], [3, 4]], self.default_dtype)
        assert tns.shape == (2, 2)

    def test_2d_ndarray_creation(self):
        tns = FlexpointTensor(np.array([[1.5, 2.5], [3.3, 9.2],
                                        [0.111111, 5]]),
                              self.default_dtype)
        assert tns.shape == (3, 2)

    def test_str(self):
        tns = FlexpointTensor([[1, 2], [3, 4]], self.default_dtype)
        assert str(tns) == "[[ 1.  2.]\n [ 3.  4.]]"

    def test_nofrac_trunc(self):
        dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=0,
                             overflow=0, rounding=0)
        tns = FlexpointTensor([[1.8, 2.1], [-3.2, -4.5]], dtype)
        assert str(tns) == "[[ 1.  2.]\n [-3. -4.]]"

    def test_nofrac_round(self):
        dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=0,
                             overflow=0, rounding=2)
        tns = FlexpointTensor([[1.8, 2.1, 2.5], [-3.2, -4.5, -3.6]], dtype)
        assert str(tns) == "[[ 2.  2.  3.]\n [-3. -5. -4.]]"

    def test_overflow_sat(self):
        dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=2,
                             overflow=0, rounding=0)
        tns = FlexpointTensor([[15.0, 15.75, 16.0, 239.3],
                               [-15.0, -16.0, -16.12, -432]], dtype)
        assert str(tns) == ("[[ 15.    15.75  15.75  15.75]\n"
                            " [-15.   -16.   -16.   -16.  ]]")

    def test_underflow_sat(self):
        dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=2,
                             overflow=0, rounding=0)
        tns = FlexpointTensor([[0.25, 0.15], [-0.25, -0.15]], dtype)
        assert str(tns) == ("[[ 0.25  0.  ]\n [-0.25  0.  ]]")
