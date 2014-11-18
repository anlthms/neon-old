# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.transforms.rectified import rectlin, rectlin_derivative
from neon.util.testing import assert_tensor_equal


def test_rectlin_positives():
    assert_tensor_equal(np.array([1, 3, 2]), rectlin(np.array([1, 3, 2])))


def test_rectlin_negatives():
    assert_tensor_equal(np.array([[0, 0], [0, 0]]),
                        rectlin(np.array([[-1, -3], [-2, -4]])))


def test_rectlin_mixed():
    assert_tensor_equal(np.array([[4, 0], [0, 9]]),
                        rectlin(np.array([[4, 0], [-2, 9]])))


def test_rectlin_cputensor():
    be = CPU()
    temp = be.zeros((2, 2))
    be.rectlin(CPUTensor([[4, 0], [-2, 9]]), temp)
    assert_tensor_equal(CPUTensor([[4, 0], [0, 9]]), temp)


@attr('cuda')
def test_rectlin_gputensor():
    # TODO: fix cudanet init/shutdown then replace
    from neon.backends.unsupported._cudamat import CudamatTensor as GPUTensor
    # with:
    # from neon.backends.gpu import GPUTensor
    assert_tensor_equal(GPUTensor([[4, 0], [0, 9]]),
                        rectlin(GPUTensor([[4, 0], [-2, 9]])))


def test_rectlin_derivative_positives():
    assert_tensor_equal(np.array([1, 1, 1]),
                        rectlin_derivative(np.array([1, 3, 2])))


def test_rectlin_derivative_negatives():
    assert_tensor_equal(np.array([[0, 0], [0, 0]]),
                        rectlin_derivative(np.array([[-1, -3], [-2, -4]])))


def test_rectlin_derivative_mixed():
    assert_tensor_equal(np.array([[1, 0], [0, 1]]),
                        rectlin_derivative(np.array([[4, 0], [-2, 9]])))


def test_rectlin_derivative_cputensor():
    be = CPU()
    temp = be.zeros((2, 2))
    be.rectlin_derivative(CPUTensor([[4, 0], [-2, 9]]), temp)
    assert_tensor_equal(CPUTensor([[1, 0], [0, 1]]), temp)


@attr('cuda')
def test_rectlin_derivative_gputensor():
    # TODO: fix cudanet init/shutdown then replace
    from neon.backends.unsupported._cudamat import CudamatTensor as GPUTensor
    # with:
    # from neon.backends.gpu import GPUTensor
    assert_tensor_equal(GPUTensor([[1, 0], [0, 1]]),
                        rectlin_derivative(GPUTensor([[4, 0], [-2, 9]])))
