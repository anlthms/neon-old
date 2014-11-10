#!/usr/bin/env/python

from neon.backends.cpu import CPU, CPUTensor
from neon.util.testing import assert_tensor_equal


class TestCPU(object):

    def __init__(self):
        # this code gets called prior to each test
        pass

    def test_empty_creation(self):
        be = CPU()
        tns = be.empty((4, 3))
        assert tns.shape == (4, 3)

    def test_array_creation(self):
        be = CPU()
        tns = be.array([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)
        assert_tensor_equal(tns, CPUTensor([[1, 2], [3, 4]]))

    def test_zeros_creation(self):
        be = CPU()
        tns = be.zeros([3, 1])
        assert tns.shape == (3, 1)
        assert_tensor_equal(tns, CPUTensor([[0], [0], [0]]))

    def test_ones_creation(self):
        be = CPU()
        tns = be.ones([1, 4])
        assert tns.shape == (1, 4)
        assert_tensor_equal(tns, CPUTensor([[1, 1, 1, 1]]))

    def test_all_equal(self):
        be = CPU()
        left = be.ones([2, 2])
        right = be.ones([2, 2])
        out = be.empty([2, 2])
        be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[1, 1], [1, 1]]))

    def test_some_equal(self):
        be = CPU()
        left = be.ones([2, 2])
        right = be.array([[0, 1], [0, 1]])
        out = be.empty([2, 2])
        be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[0, 1], [0, 1]]))

    def test_none_equal(self):
        be = CPU()
        left = be.ones([2, 2])
        right = be.zeros([2, 2])
        out = be.empty([2, 2])
        be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[0, 0], [0, 0]]))

    def test_all_not_equal(self):
        be = CPU()
        left = be.ones([2, 2])
        right = be.zeros([2, 2])
        out = be.empty([2, 2])
        be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[1, 1], [1, 1]]))

    def test_some_not_equal(self):
        be = CPU()
        left = be.ones([2, 2])
        right = be.array([[0, 1], [0, 1]])
        out = be.empty([2, 2])
        be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[1, 0], [1, 0]]))

    def test_none_not_equal(self):
        be = CPU()
        left = be.ones([2, 2])
        right = be.ones([2, 2])
        out = be.empty([2, 2])
        be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[0, 0], [0, 0]]))

    def test_greater(self):
        be = CPU()
        left = be.array([[-1, 0], [1, 92]])
        right = be.ones([2, 2])
        out = be.empty([2, 2])
        be.greater(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[0, 0], [0, 1]]))

    def test_greater_equal(self):
        be = CPU()
        left = be.array([[-1, 0], [1, 92]])
        right = be.ones([2, 2])
        out = be.empty([2, 2])
        be.greater_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[0, 0], [1, 1]]))

    def test_less(self):
        be = CPU()
        left = be.array([[-1, 0], [1, 92]])
        right = be.ones([2, 2])
        out = be.empty([2, 2])
        be.less(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[1, 1], [0, 0]]))

    def test_less_equal(self):
        be = CPU()
        left = be.array([[-1, 0], [1, 92]])
        right = be.ones([2, 2])
        out = be.empty([2, 2])
        be.less_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, CPUTensor([[1, 1], [1, 0]]))
