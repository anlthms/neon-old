# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.cross_entropy import (cross_entropy,
                                           cross_entropy_derivative)
from neon.util.testing import assert_tensor_near_equal


def test_cross_entropy_cputensor():
    be = CPU(rng_seed=0)
    outputs = CPUTensor([0.5, 0.9, 0.1, 0.0001])
    targets = CPUTensor([0.5, 0.99, 0.01, 0.2])
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = np.sum((- targets.asnumpyarray()) *
                             np.log(outputs.asnumpyarray()) -
                             (1 - targets.asnumpyarray()) *
                             np.log(1 - outputs.asnumpyarray()), keepdims=True)
    assert_tensor_near_equal(expected_result, cross_entropy(be, outputs,
                                                            targets, temp))


@attr('cuda')
def test_cross_entropy_cc2tensor():
    from neon.backends.cc2 import GPU, GPUTensor
    be = GPU(rng_seed=0)  # to ensure cublas_init() is called.
    outputs = GPUTensor([0.5, 0.9, 0.1, 0.0001])
    targets = GPUTensor([0.5, 0.99, 0.01, 0.2])
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = np.sum((- targets.asnumpyarray()) *
                             np.log(outputs.asnumpyarray()) -
                             (1 - targets.asnumpyarray()) *
                             np.log(1 - outputs.asnumpyarray()), keepdims=True)
    assert_tensor_near_equal(expected_result, cross_entropy(be, outputs,
                                                            targets, temp),
                             tolerance=1e-6)


def test_cross_entropy_derivative_cputensor():
    be = CPU(rng_seed=0)
    outputs = CPUTensor([0.5, 0.9, 0.1, 0.0001])
    targets = CPUTensor([0.5, 0.99, 0.01, 0.2])
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = ((outputs.asnumpyarray() - targets.asnumpyarray()) /
                       (outputs.asnumpyarray() * (1 - outputs.asnumpyarray())))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(be, outputs,
                                                      targets, temp))


@attr('cuda')
def test_cross_entropy_derivative_cc2tensor():
    from neon.backends.cc2 import GPU, GPUTensor
    be = GPU(rng_seed=0)
    outputs = GPUTensor([0.5, 0.9, 0.1, 0.0001])
    targets = GPUTensor([0.5, 0.99, 0.01, 0.2])
    temp = [be.zeros(outputs.shape), be.zeros(outputs.shape)]
    expected_result = ((outputs.asnumpyarray() - targets.asnumpyarray()) /
                       (outputs.asnumpyarray() * (1 - outputs.asnumpyarray())))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(be, outputs,
                                                      targets, temp))
