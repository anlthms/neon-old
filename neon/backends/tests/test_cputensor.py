#!/usr/bin/env/python

import numpy as np

from neon.backends.cpu import CPUTensor
from neon.util.testing import assert_tensor_equal


class TestCPUTensor(object):

    def __init__(self):
        # this code gets called prior to each test
        pass

    def test_empty_creation(self):
        tns = CPUTensor([])
        assert tns.shape == (0, )

    def test_1d_creation(self):
        tns = CPUTensor([1, 2, 3, 4])
        assert tns.shape == (4, )

    def test_2d_creation(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)

    def test_2d_ndarray_creation(self):
        tns = CPUTensor(np.array([[1.5, 2.5], [3.3, 9.2],
                                  [0.111111, 5]]))
        assert tns.shape == (3, 2)

    def test_higher_dim_creation(self):
        shapes = ((1, 1, 1), (1, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7))
        for shape in shapes:
            tns = CPUTensor(np.empty(shape))
            assert tns.shape == shape

    def test_str(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        assert str(tns) == "[[ 1.  2.]\n [ 3.  4.]]"

    def test_scalar_slicing(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        res = tns[1, 0]
        assert res.shape == ()  # this matches numpy behavior
        assert_tensor_equal(res, CPUTensor(3))

    def test_range_slicing(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        res = tns[0:2, 0]
        assert res.shape == (2, )
        assert_tensor_equal(res, CPUTensor([1, 3]))

    def test_scalar_slice_assignment(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        tns[1, 0] = 9
        assert_tensor_equal(tns, CPUTensor([[1, 2], [9, 4]]))

    def test_asnumpyarray(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        res = tns.asnumpyarray()
        assert isinstance(res, np.ndarray)
        assert_tensor_equal(res, np.array([[1, 2], [3, 4]]))

    def test_transpose(self):
        tns = CPUTensor([[1, 2], [3, 4]])
        res = tns.transpose()
        assert_tensor_equal(res, CPUTensor([[1, 3], [2, 4]]))