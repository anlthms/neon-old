# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from math import tanh as true_tanh

from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.tanh import Tanh
from neon.util.testing import assert_tensor_near_equal


def test_tanh_cputensor():
    tntest = Tanh()
    be = CPU(rng_seed=0)
    CPUTensor([0, 1, -2])
    inputs = np.array([0, 1, -2])
    temp = be.zeros(inputs.shape)
    outputs = np.array([true_tanh(0), true_tanh(1), true_tanh(-2)])
    tntest.apply_function(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_tanh_gputensor():
    tntest = Tanh()
    from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = GPUTensor([true_tanh(0), true_tanh(1), true_tanh(-2)])
    be = GPU(rng_seed=0)
    temp = be.zeros((3, 1))
    tntest.apply_function(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(outputs, temp)


def test_tanh_derivative_cputensor():
    tntest = Tanh()
    inputs = np.array([0, 1, -2])
    be = CPU(rng_seed=0)
    outputs = np.array([1 - true_tanh(0) ** 2,
                        1 - true_tanh(1) ** 2,
                        1 - true_tanh(-2) ** 2])
    temp = be.zeros(inputs.shape)
    tntest.apply_derivative(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_tanh_derivative_gputensor():
    tntest = Tanh()
    from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2], dtype='float32').reshape((3, 1))
    be = GPU(rng_seed=0)
    outputs = GPUTensor([1 - true_tanh(0) ** 2,
                         1 - true_tanh(1) ** 2,
                         1 - true_tanh(-2) ** 2])
    temp = be.zeros(inputs.shape)
    tntest.apply_derivative(be, GPUTensor(inputs, dtype='float32'), temp)
    assert_tensor_near_equal(outputs, temp, tolerance=1e-5)
