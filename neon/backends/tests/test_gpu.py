#!/usr/bin/env/python

from nose.plugins.attrib import attr
from nose.tools import nottest

from neon.util.compat import CUDA_GPU
from neon.util.testing import assert_tensor_equal

if CUDA_GPU:
    # TODO: resolve.  Currently conflicts with cuda RBM tests (can only
    # instantiate once during make test run)
    # from neon.backends.gpu import GPU, GPUTensor
    GPUTensor = None
    pass


class TestGPU(object):

    @attr('cuda')
    def __init__(self):
        # this code gets called prior to each test
        # TODO: resolve multiple GPU() call
        # self.be = GPU(rng_seed=0)
        self.be = None


    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_empty_creation(self):
        tns = self.be.empty((4, 3))
        assert tns.shape == (4, 3)

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_array_creation(self):
        tns = self.be.array([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)
        assert_tensor_equal(tns, GPUTensor([[1, 2], [3, 4]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_zeros_creation(self):
        tns = self.be.zeros([3, 1])
        assert tns.shape == (3, 1)
        assert_tensor_equal(tns, GPUTensor([[0], [0], [0]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_ones_creation(self):
        tns = self.be.ones([1, 4])
        assert tns.shape == (1, 4)
        assert_tensor_equal(tns, GPUTensor([[1, 1, 1, 1]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_all_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[1, 1], [1, 1]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_some_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.array([[0, 1], [0, 1]])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[0, 1], [0, 1]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_none_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.zeros([2, 2])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[0, 0], [0, 0]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_all_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.zeros([2, 2])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[1, 1], [1, 1]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_some_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.array([[0, 1], [0, 1]])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[1, 0], [1, 0]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_none_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[0, 0], [0, 0]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_greater(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.greater(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[0, 0], [0, 1]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_greater_equal(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.greater_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[0, 0], [1, 1]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_less(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.less(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[1, 1], [0, 0]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_less_equal(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.less_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, GPUTensor([[1, 1], [1, 0]]))
