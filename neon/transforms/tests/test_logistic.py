# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.logistic import logistic, logistic_derivative
from neon.util.testing import assert_tensor_near_equal


def test_logistic_cputensor():
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    temp = be.zeros((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    logistic(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_logistic_gputensor():
    from neon.backends.unsupported._cudamat import (Cudamat as GPU,  # flake8:noqa
                                                    CudamatTensor as GPUTensor)
    # with:
    # from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    be = GPU(rng_seed=0)
    temp = be.zeros((3, 1))
    logistic(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)


def test_logistic_derivative_cputensor():
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    outputs = outputs * (1.0 - outputs)
    temp = be.zeros(inputs.shape)
    logistic_derivative(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_logistic_derivative_gputensor():
    # TODO: fix cudanet init/shutdown then replace
    from neon.backends.unsupported._cudamat import (Cudamat as GPU,
                                                    CudamatTensor as GPUTensor)
    # with:
    # from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    outputs = outputs * (1.0 - outputs)
    be = GPU(rng_seed=0)
    temp = be.zeros(inputs.shape)
    logistic_derivative(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)
