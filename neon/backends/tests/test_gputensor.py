#!/usr/bin/env/python

from nose.plugins.attrib import attr
from nose.tools import nottest
import numpy as np

from neon.util.testing import assert_tensor_equal
from neon.util.compat import CUDA_GPU

if CUDA_GPU:
    from neon.backends.gpu import GPUTensor


class TestGPUTensor(object):

    @attr('cuda')
    @nottest  # TODO: fix the empty shape
    def test_empty_creation(self):
        tns = GPUTensor([])
        assert tns.shape == (1, 1)

    @attr('cuda')
    def test_1d_creation(self):
        tns = GPUTensor([1, 2, 3, 4])
        assert tns.shape == (4, 1)

    @attr('cuda')
    def test_2d_creation(self):
        tns = GPUTensor([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)

    @attr('cuda')
    def test_2d_ndarray_creation(self):
        tns = GPUTensor(np.array([[1.5, 2.5], [3.3, 9.2],
                                  [0.111111, 5]]))
        assert tns.shape == (3, 2)

    @attr('cuda')
    @nottest  # TODO: add >2 dimension support to cudanet
    def test_higher_dim_creation(self):
        shapes = ((1, 1, 1), (1, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7))
        for shape in shapes:
            tns = GPUTensor(np.empty(shape))
            assert tns.shape == shape

    @attr('cuda')
    def test_str(self):
        tns = GPUTensor([[1, 2], [3, 4]])
        assert str(tns) == "[[1 2]\n [3 4]]"

    @attr('cuda')
    @nottest  # TODO: fix this comparison
    def test_scalar_slicing(self):
        tns = GPUTensor([[1, 2], [3, 4]])
        res = tns[1, 0]
        assert res.shape == (1, 1)
        assert_tensor_equal(res, GPUTensor(3))

    @attr('cuda')
    @nottest  # TODO: fix this comparison
    def test_range_slicing(self):
        tns = GPUTensor([[1, 2], [3, 4]])
        res = tns[0:2, 0]
        assert res.shape == (2, 1)
        assert_tensor_equal(res, GPUTensor([1, 3]))

    @attr('cuda')
    @nottest  # TODO: add scalar assignment to GPUTensor class
    def test_scalar_slice_assignment(self):
        tns = GPUTensor([[1, 2], [3, 4]])
        tns[1, 0] = 9
        assert_tensor_equal(tns, GPUTensor([[1, 2], [9, 4]]))

    @attr('cuda')
    def test_asnumpyarray(self):
        tns = GPUTensor([[1, 2], [3, 4]])
        res = tns.asnumpyarray()
        assert isinstance(res, np.ndarray)
        assert_tensor_equal(res, np.array([[1, 2], [3, 4]]))

    # @attr('cuda')
    @nottest  # TODO: fix this for GPUTensor
    def test_transpose(self):
        tns = GPUTensor([[1, 2], [3, 4]])
        res = tns.transpose()
        assert_tensor_equal(res, GPUTensor([[1, 3], [2, 4]]))
