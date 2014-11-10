#!/usr/bin/env/python

import numpy as np

from neon.backends.flexpoint import (Flexpoint, FlexpointTensor, flexpt_dtype)
from neon.util.testing import assert_tensor_equal


class TestFlexpoint(object):

    def __init__(self):
        # this code gets called prior to each test
        self.default_dtype = flexpt_dtype(sign_bit=True, int_bits=4,
                                          frac_bits=11, overflow=0, rounding=0)
        self.be = Flexpoint()

    def test_empty_creation(self):
        tns = self.be.empty((4, 3))
        assert tns.shape == (4, 3)

    def test_array_creation(self):
        tns = self.be.array([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)
        assert_tensor_equal(tns, FlexpointTensor([[1, 2], [3, 4]]))

    def test_zeros_creation(self):
        tns = self.be.zeros([3, 1])
        assert tns.shape == (3, 1)
        assert_tensor_equal(tns, FlexpointTensor([[0], [0], [0]]))

    def test_ones_creation(self):
        tns = self.be.ones([1, 4])
        assert tns.shape == (1, 4)
        assert_tensor_equal(tns, FlexpointTensor([[1, 1, 1, 1]]))

    def test_all_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[1, 1], [1, 1]]))

    def test_some_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.array([[0, 1], [0, 1]])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[0, 1], [0, 1]]))

    def test_none_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.zeros([2, 2])
        out = self.be.empty([2, 2])
        self.be.equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[0, 0], [0, 0]]))

    def test_all_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.zeros([2, 2])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[1, 1], [1, 1]]))

    def test_some_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.array([[0, 1], [0, 1]])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[1, 0], [1, 0]]))

    def test_none_not_equal(self):
        left = self.be.ones([2, 2])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.not_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[0, 0], [0, 0]]))

    def test_greater(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.greater(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[0, 0], [0, 1]]))

    def test_greater_equal(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.greater_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[0, 0], [1, 1]]))

    def test_less(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.less(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[1, 1], [0, 0]]))

    def test_less_equal(self):
        left = self.be.array([[-1, 0], [1, 92]])
        right = self.be.ones([2, 2])
        out = self.be.empty([2, 2])
        self.be.less_equal(left, right, out)
        assert out.shape == (2, 2)
        assert_tensor_equal(out, FlexpointTensor([[1, 1], [1, 0]]))
