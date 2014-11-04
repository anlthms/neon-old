from math import tanh as true_tanh

from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPUTensor
from neon.transforms.tanh import tanh, tanh_derivative
from neon.util.testing import assert_tensor_near_equal


def test_tanh_basics():
    assert_tensor_near_equal(np.array([true_tanh(0), true_tanh(1),
                                       true_tanh(-2)]),
                             tanh(np.array([0, 1, -2])))


def test_tanh_cputensor():
    assert_tensor_near_equal(CPUTensor([true_tanh(0), true_tanh(1),
                                        true_tanh(-2)]),
                             tanh(CPUTensor([0, 1, -2])))


@attr('cuda')
def test_tanh_cudamattensor():
    from neon.backends.gpu import GPUTensor
    assert_tensor_near_equal(GPUTensor([true_tanh(0), true_tanh(1),
                                        true_tanh(-2)]),
                             tanh(GPUTensor([0, 1, -2])))


def test_tanh_derivative_basics():
    assert_tensor_near_equal(np.array([1 - true_tanh(0) ** 2,
                                       1 - true_tanh(1) ** 2,
                                       1 - true_tanh(-2) ** 2]),
                             tanh_derivative(np.array([0, 1, -2])))


def test_tanh_derivative_cputensor():
    assert_tensor_near_equal(CPUTensor([1 - true_tanh(0) ** 2,
                                        1 - true_tanh(1) ** 2,
                                        1 - true_tanh(-2) ** 2]),
                             tanh_derivative(CPUTensor([0, 1, -2])))


@attr('cuda')
def test_tanh_derivative_gputensor():
    from neon.backends.gpu import GPUTensor
    assert_tensor_near_equal(GPUTensor([1 - true_tanh(0) ** 2,
                                        1 - true_tanh(1) ** 2,
                                        1 - true_tanh(-2) ** 2]),
                             tanh_derivative(GPUTensor([0, 1, -2])))
