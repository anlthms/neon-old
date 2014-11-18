#!/usr/bin/env/python

from nose.tools import nottest
import numpy as np

from neon.backends.flexpoint import (Flexpoint, FlexpointTensor, flexpt_dtype)
from neon.util.testing import assert_tensor_equal, assert_tensor_near_equal


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

    @nottest  # TODO: fix dimension handling
    def test_argmin_noaxis(self):
        be = Flexpoint()
        tsr = be.array([[-1, 0], [1, 92]])
        out = be.empty([1, 1])
        be.argmin(tsr, None, out)
        assert_tensor_equal(out, FlexpointTensor([[0]]))

    @nottest  # TODO: fix dimension handling
    def test_argmin_axis0(self):
        be = Flexpoint()
        tsr = be.array([[-1, 0], [1, 92]])
        out = be.empty((2, ))
        be.argmin(tsr, 0, out)
        assert_tensor_equal(out, FlexpointTensor([0, 0]))

    @nottest  # TODO: fix dimension handling
    def test_argmin_axis1(self):
        be = Flexpoint()
        tsr = be.array([[-1, 10], [11, 9]])
        out = be.empty((2, ))
        be.argmin(tsr, 1, out)
        assert_tensor_equal(out, FlexpointTensor([0, 1]))

    @nottest  # TODO: fix dimension handling
    def test_argmax_noaxis(self):
        be = Flexpoint()
        tsr = be.array([[-1, 0], [1, 92]])
        out = be.empty([1, 1])
        be.argmax(tsr, None, out)
        assert_tensor_equal(out, FlexpointTensor(3))

    @nottest  # TODO: fix dimension handling
    def test_argmax_axis0(self):
        be = Flexpoint()
        tsr = be.array([[-1, 0], [1, 92]])
        out = be.empty((2, ))
        be.argmax(tsr, 0, out)
        assert_tensor_equal(out, FlexpointTensor([1, 1]))

    @nottest  # TODO: fix dimension handling
    def test_argmax_axis1(self):
        be = Flexpoint()
        tsr = be.array([[-1, 10], [11, 9]])
        out = be.empty((2, ))
        be.argmax(tsr, 1, out)
        assert_tensor_equal(out, FlexpointTensor([1, 0]))

    def test_2norm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        rpow = 1. / 2
        # -> sum([[1, 0], [1, 9]], axis=0)**.5 -> sqrt([2, 9])
        assert_tensor_equal(self.be.norm(tsr, order=2, axis=0),
                            FlexpointTensor([2**rpow, 9**rpow]))
        # -> sum([[1, 0], [1, 9]], axis=1)**.5 -> sqrt([1, 10])
        assert_tensor_equal(self.be.norm(tsr, order=2, axis=1),
                            FlexpointTensor([1**rpow, 10**rpow]))

    def test_1norm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        # -> sum([[1, 0], [1, 3]], axis=0)**1 -> [2, 3]
        assert_tensor_equal(self.be.norm(tsr, order=1, axis=0),
                            FlexpointTensor([2, 3]))
        # -> sum([[1, 0], [1, 3]], axis=1)**1 -> [1, 4]
        assert_tensor_equal(self.be.norm(tsr, order=1, axis=1),
                            FlexpointTensor([1, 4]))

    def test_0norm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        # -> sum(tsr != 0, axis=0) -> [2, 1]
        assert_tensor_equal(self.be.norm(tsr, order=0, axis=0),
                            FlexpointTensor([2, 1]))
        # -> sum(tsr != 0, axis=1) -> [1, 2]
        assert_tensor_equal(self.be.norm(tsr, order=0, axis=1),
                            FlexpointTensor([1, 2]))

    def test_infnorm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        # -> max(abs(tsr), axis=0) -> [1, 3]
        assert_tensor_equal(self.be.norm(tsr, order=float('inf'), axis=0),
                            FlexpointTensor([1, 3]))
        # -> max(abs(tsr), axis=1) -> [1, 3]
        assert_tensor_equal(self.be.norm(tsr, order=float('inf'), axis=1),
                            FlexpointTensor([1, 3]))

    def test_neginfnorm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        # -> min(abs(tsr), axis=0) -> [1, 0]
        assert_tensor_equal(self.be.norm(tsr, order=float('-inf'), axis=0),
                            FlexpointTensor([1, 0]))
        # -> min(abs(tsr), axis=1) -> [0, 1]
        assert_tensor_equal(self.be.norm(tsr, order=float('-inf'), axis=1),
                            FlexpointTensor([0, 1]))

    def test_lrgnorm(self):
        tsr = self.be.array([[-1, 0], [1, 3]])
        rpow = 1. / 5
        # -> sum([[1, 0], [1, 243]], axis=0)**rpow -> rpow([2, 243])
        assert_tensor_equal(self.be.norm(tsr, order=5, axis=0),
                            FlexpointTensor([2**rpow, 243**rpow]))
        # -> sum([[1, 0], [1, 243]], axis=1)**rpow -> rpow([1, 244])
        # 244**.2 == ~3.002465 hence the near_equal test
        assert_tensor_near_equal(self.be.norm(tsr, order=5, axis=1),
                                 FlexpointTensor([1**rpow, 244**rpow]), 1e-6)

    def test_negnorm(self):
        tsr = self.be.array([[-1, -2], [1, 3]])
        rpow = -1. / 3
        # -> sum([[1, .125], [1, .037037]], axis=0)**rpow -> rpow([2, .162037])
        assert_tensor_equal(self.be.norm(tsr, order=-3, axis=0),
                            FlexpointTensor([2**rpow, .162037037037**rpow]))
        # -> sum([[1, .125], [1, .037037]], axis=1)**rpow ->
        # rpow([1.125, 1.037037])
        assert_tensor_near_equal(self.be.norm(tsr, order=-3, axis=1),
                                 FlexpointTensor([1.125**rpow,
                                                  1.037037**rpow]),
                                 1e-6)
