# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.softmax import Softmax
from neon.util.testing import assert_tensor_near_equal


def test_softmax_cputensor():
    sftmx = Softmax()
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    temp = be.zeros((3, 1))
    outputs = np.exp(inputs-1) / np.sum(np.exp(inputs-1))
    sftmx.apply_function(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_softmax_gputensor():
    sftmx = Softmax()
    from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.exp(inputs) / np.sum(np.exp(inputs))
    be = GPU(rng_seed=0)
    temp = be.zeros((3, 1))
    sftmx.apply_function(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)


def test_softmax_derivative_cputensor():
    sftmx = Softmax()
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    be = CPU(rng_seed=0)
    outputs = np.exp(inputs) / np.sum(np.exp(inputs))
    errmat = np.ones(inputs.shape)
    a = np.einsum('ij,ji->i', errmat.T, outputs)
    outputs = outputs * (errmat - a[np.newaxis, :])
    temp = be.zeros(inputs.shape)
    sftmx.apply_derivative(be, CPUTensor(inputs), temp)
    assert_tensor_near_equal(CPUTensor(outputs), temp)


@attr('cuda')
def test_softmax_derivative_gputensor():
    sftmx = Softmax()
    from neon.backends.gpu import GPU, GPUTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = np.exp(inputs) / np.sum(np.exp(inputs))
    errmat = np.ones(inputs.shape)
    a = np.einsum('ij,ji->i', errmat.T, outputs)
    outputs = outputs * (errmat - a[np.newaxis, :])
    be = GPU(rng_seed=0)
    temp = be.zeros(inputs.shape)
    sftmx.apply_derivative(be, GPUTensor(inputs), temp)
    assert_tensor_near_equal(GPUTensor(outputs), temp)
