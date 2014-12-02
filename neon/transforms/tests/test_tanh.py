# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.tanh import tanh, tanh_derivative
from neon.util.testing import assert_tensor_near_equal

from numpy import tanh as true_tanh


def test_tanh_cputensor():
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    temp = be.zeros((3, 1))
    outputs = true_tanh(inputs)
    tanh(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_tanh_gputensor():
    from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = true_tanh(inputs)
    be = GPU(rng_seed=0)
    temp = be.zeros((3, 1))
    tanh(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)


def test_tanh_derivative_cputensor():
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    outputs = 1.0 - true_tanh(inputs)**2
    temp1 = be.zeros(inputs.shape)
    temp2 = be.zeros(inputs.shape)
    tanh(be, CPUTensor(inputs), temp1)
    tanh_derivative(be, temp1, temp2)
    assert_tensor_near_equal(CPUTensor(outputs), temp2)


@attr('cuda')
def test_tanh_derivative_gputensor():
    from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 - true_tanh(inputs)**2
    be = GPU(rng_seed=0)
    temp1 = be.zeros(inputs.shape)
    temp2 = be.zeros(inputs.shape)
    tanh(be, GPUTensor(inputs), temp1)
    tanh_derivative(be, temp1, temp2)
    assert_tensor_near_equal(GPUTensor(outputs), temp2)
