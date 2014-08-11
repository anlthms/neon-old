from nose.plugins.attrib import attr
import numpy as np

from mylearn.backends._numpy import NumpyTensor
from mylearn.transforms.rectified import rectlin, rectlin_derivative
from mylearn.util.testing import assert_tensor_equal


def test_rectlin_positives():
    assert_tensor_equal(np.array([1, 3, 2]), rectlin(np.array([1, 3, 2])))


def test_rectlin_negatives():
    assert_tensor_equal(np.array([[0, 0], [0, 0]]),
                        rectlin(np.array([[-1, -3], [-2, -4]])))


def test_rectlin_mixed():
    assert_tensor_equal(np.array([[4, 0], [0, 9]]),
                        rectlin(np.array([[4, 0], [-2, 9]])))


def test_rectlin_NumpyTensor():
    assert_tensor_equal(NumpyTensor([[4, 0], [0, 9]]),
                        rectlin(NumpyTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_CudamatTensor():
    from mylearn.backends._cudamat import CudamatTensor
    assert_tensor_equal(CudamatTensor([[4, 0], [0, 9]]),
                        rectlin(CudamatTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_diff_backends():
    from mylearn.backends._cudamat import CudamatTensor
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


def test_rectlin_derivative_NumpyTensor():
    assert_tensor_equal(NumpyTensor([[1, 0], [0, 1]]),
                        rectlin_derivative(NumpyTensor([[4, 0], [-2, 9]])))


@attr('cuda')
def test_rectlin_derivative_CudamatTensor():
    from mylearn.backends._cudamat import CudamatTensor
    assert_tensor_equal(CudamatTensor([[1, 0], [0, 1]]),
                        rectlin_derivative(CudamatTensor([[4, 0], [-2, 9]])))


def test_rectlin_derivative_diff_backends():
    assert_tensor_equal(np.array([[1, 0], [0, 1]]),
                        rectlin_derivative(NumpyTensor([[4, 0], [-2, 9]])))
