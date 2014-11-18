#!/usr/bin/env/python

from nose.tools import nottest
import numpy as np

from neon.backends.flexpoint import (FlexpointTensor, flexpt_dtype)
from neon.util.testing import assert_tensor_equal


class TestFlexpointTensor(object):

    def __init__(self):
        # this code gets called prior to each test
        self.default_dtype = flexpt_dtype(sign_bit=True, int_bits=10,
                                          frac_bits=5, overflow=0, rounding=0)

    @nottest  # TODO: fix the empty shape
    def test_empty_creation(self):
        tns = FlexpointTensor([])
        assert tns.shape == (0, )

    def test_1d_creation(self):
        tns = FlexpointTensor([1, 2, 3, 4])
        assert tns.shape == (4, 1)

    def test_2d_creation(self):
        tns = FlexpointTensor([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)

    def test_2d_ndarray_creation(self):
        tns = FlexpointTensor(np.array([[1.5, 2.5], [3.3, 9.2],
                                        [0.111111, 5]]))
        assert tns.shape == (3, 2)

    @nottest  # TODO: support n-dimensional arrays
    def test_higher_dim_creation(self):
        shapes = ((1, 1, 1), (1, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7))
        for shape in shapes:
            tns = FlexpointTensor(np.empty(shape))
            assert tns.shape == shape

    def test_str(self):
        tns = FlexpointTensor([[1, 2], [3, 4]])
        assert str(tns) == "[[ 1.  2.]\n [ 3.  4.]]"

    def test_scalar_slicing(self):
        tns = FlexpointTensor([[1, 2], [3, 4]])
        res = tns[1, 0]
        assert res.shape == (1, 1)
        assert_tensor_equal(res, FlexpointTensor(3))

    def test_range_slicing(self):
        tns = FlexpointTensor([[1, 2], [3, 4]])
        res = tns[0:2, 0]
        assert res.shape == (2, 1)
        assert_tensor_equal(res, FlexpointTensor([1, 3]))

    def test_scalar_slice_assignment(self):
        tns = FlexpointTensor([[1, 2], [3, 4]])
        tns[1, 0] = 9.0
        assert_tensor_equal(tns, FlexpointTensor([[1, 2], [9, 4]]))

    def test_asnumpyarray(self):
        tns = FlexpointTensor([[1, 2], [3, 4]])
        res = tns.asnumpyarray()
        assert isinstance(res, np.ndarray)
        assert_tensor_equal(res, np.array([[1, 2], [3, 4]]))

    def test_transpose(self):
        tns = FlexpointTensor([[1, 2], [3, 4]])
        res = tns.transpose()
        assert_tensor_equal(res, FlexpointTensor([[1, 3], [2, 4]]))

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
