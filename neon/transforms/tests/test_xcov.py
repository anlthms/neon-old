# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.xcov import (xcov_cost, xcov_cost_derivative)
from neon.util.testing import assert_tensor_near_equal


def xcc(x, y):
    return (x - x.mean(axis=1, keepdims=True)).dot(
        (y - y.mean(axis=1, keepdims=True)).T)/x.shape[1]


def test_xcov_cputensor():
    np.random.seed(0)
    n = 10
    k = 8
    (k1, k2) = (3, 5)
    a = np.array(np.random.randn(k, n)*10, dtype='float32', order='C')
    acc = xcc(a[:k1], a[k1:])
    expected_result = 0.5 * (acc**2.).sum()

    be = CPU(rng_seed=0)
    outputs = CPUTensor(be.copy(a))
    tempbuf1 = be.empty((k1, n))
    tempbuf2 = be.empty((k2, n))
    tempbuf3 = be.empty((k1, k2))
    tempbuf4 = be.empty(outputs.shape)
    temp = [tempbuf1, tempbuf2, tempbuf3, tempbuf4]
    my_result = xcov_cost(be, outputs, [], temp, k1)
    assert_tensor_near_equal(expected_result, my_result)


@attr('cuda')
def test_xcov_gputensor():
    np.random.seed(0)
    n = 10
    k = 8
    (k1, k2) = (3, 5)
    a = np.array(np.random.randn(k, n)*10, dtype='float32', order='C')
    acc = xcc(a[:k1], a[k1:])
    expected_result = 0.5 * (acc**2.).sum()

    from neon.backends.gpu import GPU, GPUTensor
    be = GPU(rng_seed=0)  # to ensure cublas_init() is called.
    outputs = GPUTensor(be.copy(a))
    tempbuf1 = be.empty((k1, n))
    tempbuf2 = be.empty((k2, n))
    tempbuf3 = be.empty((k1, k2))
    tempbuf4 = be.empty(outputs.shape)
    temp = [tempbuf1, tempbuf2, tempbuf3, tempbuf4]
    my_result = xcov_cost(be, outputs, [], temp, k1)
    assert_tensor_near_equal(expected_result, my_result)


def test_xcov_derivative_cputensor():
    np.random.seed(0)
    n = 10
    k = 8
    (k1, k2) = (3, 5)
    a = np.array(np.random.randn(k, n), dtype='float32', order='C')
    s = np.zeros_like(a)
    acc = xcc(a[:k1], a[k1:])  # k1 x k2
    c1 = a[k1:] - a[k1:].mean(1, keepdims=True)  # k2 x n
    c2 = a[:k1] - a[:k1].mean(1, keepdims=True)  # k1 x n

    s[:k1] = acc.dot(c1)/n
    s[k1:] = acc.T.dot(c2)/n

    be = CPU(rng_seed=0)
    outputs = CPUTensor(be.copy(a))
    tempbuf1 = be.empty((k1, n))
    tempbuf2 = be.empty((k2, n))
    tempbuf3 = be.empty((k1, k2))
    tempbuf4 = be.empty(outputs.shape)
    temp = [tempbuf1, tempbuf2, tempbuf3, tempbuf4]
    my_result = xcov_cost_derivative(be, outputs, [], temp, k1)
    expected_result = CPUTensor(s)
    assert_tensor_near_equal(expected_result, my_result)


@attr('cuda')
def test_xcov_derivative_gputensor():
    from neon.backends.gpu import GPU, GPUTensor
    be = GPU(rng_seed=0)
    np.random.seed(0)
    n = 10
    k = 8
    (k1, k2) = (3, 5)
    a = np.array(np.random.randn(k, n), dtype='float32', order='C')
    s = np.zeros_like(a)
    acc = xcc(a[:k1], a[k1:])  # k1 x k2
    c1 = a[k1:] - a[k1:].mean(1, keepdims=True)  # k2 x n
    c2 = a[:k1] - a[:k1].mean(1, keepdims=True)  # k1 x n

    s[:k1] = acc.dot(c1)/n
    s[k1:] = acc.T.dot(c2)/n

    outputs = GPUTensor(be.copy(a))
    tempbuf1 = be.empty((k1, n))
    tempbuf2 = be.empty((k2, n))
    tempbuf3 = be.empty((k1, k2))
    tempbuf4 = be.empty(outputs.shape)
    temp = [tempbuf1, tempbuf2, tempbuf3, tempbuf4]
    my_result = xcov_cost_derivative(be, outputs, [], temp, k1)
    expected_result = GPUTensor(s)
    assert_tensor_near_equal(expected_result, my_result)
