from math import tanh as true_tanh

from nose.plugins.attrib import attr
import numpy as np

from mylearn.backends._numpy import NumpyTensor
from mylearn.transforms.tanh import tanh, tanh_derivative
from mylearn.util.testing import assert_tensor_near_equal


def test_tanh_basics():
    assert_tensor_near_equal(np.array([true_tanh(0), true_tanh(1),
                             true_tanh(-2)]),
                             tanh(np.array([0, 1, -2])))

def test_tanh_NumpyTensor():
    assert_tensor_near_equal(NumpyTensor([true_tanh(0), true_tanh(1),
                             true_tanh(-2)]),
                             tanh(NumpyTensor([0, 1, -2])))

def test_tanh_derivative_basics():
    assert_tensor_near_equal(np.array([1 - true_tanh(0) ** 2,
                                       1 - true_tanh(1) ** 2,
                                       1 - true_tanh(-2) ** 2]),
                             tanh_derivative(np.array([0, 1, -2])))
