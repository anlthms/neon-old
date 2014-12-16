# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.logistic import Logistic
from neon.util.testing import assert_tensor_near_equal


def test_logistic_cputensor():
    lgstc = Logistic()
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    temp = be.zeros((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    lgstc.apply_function(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_logistic_gputensor():
    lgstc = Logistic()
    from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    be = GPU(rng_seed=0)
    temp = be.zeros((3, 1))
    lgstc.apply_function(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)


def test_logistic_derivative_cputensor():
    lgstc = Logistic()
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    outputs = outputs * (1.0 - outputs)
    temp = be.zeros(inputs.shape)
    lgstc.apply_derivative(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_logistic_derivative_gputensor():
    lgstc = Logistic()
    from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    outputs = outputs * (1.0 - outputs)
    be = GPU(rng_seed=0)
    temp = be.zeros(inputs.shape)
    lgstc.apply_derivative(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)
