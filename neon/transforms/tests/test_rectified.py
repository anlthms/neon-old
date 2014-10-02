from nose.plugins.attrib import attr
import numpy as np

from neon.backends._numpy import NumpyTensor
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


def test_rectlin_numpytensor():
    assert_tensor_equal(NumpyTensor([[4, 0], [0, 9]]),
                        rectlin(NumpyTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_cudamattensor():
    from neon.backends._cudamat import CudamatTensor
    assert_tensor_equal(CudamatTensor([[4, 0], [0, 9]]),
                        rectlin(CudamatTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_diff_backends():
    from neon.backends._cudamat import CudamatTensor
    assert_tensor_equal(NumpyTensor([[4, 0], [0, 9]]),
                        rectlin(CudamatTensor([[4, 0], [-2, 9]])))


def test_rectlin_derivative_positives():
    assert_tensor_equal(np.array([1, 1, 1]),
                        rectlin_derivative(np.array([1, 3, 2])))


def test_rectlin_derivative_negatives():
    assert_tensor_equal(np.array([[0, 0], [0, 0]]),
                        rectlin_derivative(np.array([[-1, -3], [-2, -4]])))


def test_rectlin_derivative_mixed():
    assert_tensor_equal(np.array([[1, 0], [0, 1]]),
                        rectlin_derivative(np.array([[4, 0], [-2, 9]])))


def test_rectlin_derivative_numpytensor():
    assert_tensor_equal(NumpyTensor([[1, 0], [0, 1]]),
                        rectlin_derivative(NumpyTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_derivative_cudamattensor():
    from neon.backends._cudamat import CudamatTensor
    assert_tensor_equal(CudamatTensor([[1, 0], [0, 1]]),
                        rectlin_derivative(CudamatTensor([[4, 0], [-2, 9]])))


def test_rectlin_derivative_diff_backends():
    assert_tensor_equal(np.array([[1, 0], [0, 1]]),
                        rectlin_derivative(NumpyTensor([[4, 0], [-2, 9]])))
