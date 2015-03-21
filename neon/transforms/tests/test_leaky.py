# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.rectified import RectLeaky
from neon.util.testing import assert_tensor_equal


def compare_cpu_tensors(inputs, outputs, deriv=False):
    rlin = RectLeaky()
    be = CPU()
    temp = be.zeros(inputs.shape)
    if deriv is True:
        rlin.apply_derivative(be, CPUTensor(inputs), temp)
    else:
        rlin.apply_function(be, CPUTensor(inputs), temp)
    be.subtract(temp, CPUTensor(outputs), temp)
    assert_tensor_equal(temp, be.zeros(inputs.shape))


def compare_gpu_tensors(inputs, outputs, deriv=False):
    from neon.backends.gpu import GPU, GPUTensor
    rlin = RectLeaky()
    be = GPU()
    temp = be.zeros(inputs.shape)
    if deriv is True:
        rlin.apply_derivative(be, GPUTensor(inputs), temp)
    else:
        rlin.apply_function(be, GPUTensor(inputs), temp)
    be.subtract(temp, GPUTensor(outputs), temp)
    assert_tensor_equal(temp, be.zeros(inputs.shape))


def test_rectleaky_positives():
    inputs = np.array([1, 3, 2])
    outputs = np.array([1, 3, 2])
    compare_cpu_tensors(inputs, outputs)


def test_rectleaky_negatives():
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[-0.01, -0.03], [-0.02, -0.04]])
    compare_cpu_tensors(inputs, outputs)


def test_rectleaky_mixed():
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[4, 0], [-0.02, 9]])
    compare_cpu_tensors(inputs, outputs)


@attr('cuda')
def test_rectleaky_gputensor():
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[4, 0], [-0.02, 9]])
    compare_gpu_tensors(inputs, outputs)


def test_rectleaky_derivative_positives():
    inputs = np.array([1, 3, 2])
    outputs = np.array([1, 1, 1])
    compare_cpu_tensors(inputs, outputs, deriv=True)


def test_rectleaky_derivative_negatives():
    inputs = np.array([[-1, -3], [-2, -4]])
    outputs = np.array([[0.01, 0.01], [0.01, 0.01]])
    compare_cpu_tensors(inputs, outputs, deriv=True)


def test_rectleaky_derivative_mixed():
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[1, 0.01], [0.01, 1]])
    compare_cpu_tensors(inputs, outputs, deriv=True)


@attr('cuda')
def test_rectleaky_derivative_gputensor():
    inputs = np.array([[4, 0], [-2, 9]])
    outputs = np.array([[1, 0.01], [0.01, 1]])
    compare_gpu_tensors(inputs, outputs, deriv=True)
