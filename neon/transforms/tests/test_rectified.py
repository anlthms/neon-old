from nose.plugins.attrib import attr
import numpy as np

from neon.backends.cpu import CPUTensor
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
    assert_tensor_equal(CPUTensor([[4, 0], [0, 9]]),
                        rectlin(CPUTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_gputensor():
    from neon.backends.gpu import GPUTensor
    assert_tensor_equal(GPUTensor([[4, 0], [0, 9]]),
                        rectlin(GPUTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_diff_backends():
    from neon.backends.gpu import GPUTensor
    assert_tensor_equal(CPUTensor([[4, 0], [0, 9]]),
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
    assert_tensor_equal(CPUTensor([[1, 0], [0, 1]]),
                        rectlin_derivative(CPUTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_derivative_gputensor():
    from neon.backends.gpu import GPUTensor
    assert_tensor_equal(GPUTensor([[1, 0], [0, 1]]),
                        rectlin_derivative(GPUTensor([[4, 0], [-2, 9]])))


def test_rectlin_derivative_diff_backends():
    assert_tensor_equal(np.array([[1, 0], [0, 1]]),
                        rectlin_derivative(CPUTensor([[4, 0], [-2, 9]])))
