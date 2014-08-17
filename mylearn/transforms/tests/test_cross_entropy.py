from nose.plugins.attrib import attr
import numpy as np

from mylearn.backends._numpy import Numpy, NumpyTensor
from mylearn.transforms.cross_entropy import (cross_entropy,
                                              cross_entropy_derivative)
from mylearn.util.testing import assert_tensor_near_equal


def test_cross_entropy_basic():
    outputs = np.array([0.5, 0.9, 0.1, 0.0001])
    targets = np.array([0.5, 0.99, 0.01, 0.2])
    expected_result = np.mean((- targets) * np.log(outputs) -
                              (1 - targets) * np.log(1 - outputs))
    assert_tensor_near_equal(expected_result, cross_entropy(outputs, targets))


def test_cross_entropy_NumpyTensor():
    outputs = NumpyTensor([0.5, 0.9, 0.1, 0.0001])
    targets = NumpyTensor([0.5, 0.99, 0.01, 0.2])
    expected_result = Numpy.mean((- targets) * Numpy.log(outputs) -
                                 (1 - targets) * Numpy.log(1 - outputs))
    assert_tensor_near_equal(expected_result, cross_entropy(outputs, targets))


@attr('cuda')
def test_cross_entropy_CudamatTensor():
    from mylearn.backends._cudamat import Cudamat, CudamatTensor
    outputs = CudamatTensor([0.5, 0.9, 0.1, 0.0001])
    targets = CudamatTensor([0.5, 0.99, 0.01, 0.2])
    expected_result = Cudamat.mean((- targets) * Cudamat.log(outputs) -
                                   (1 - targets) * Cudamat.log(1 - outputs))
    assert_tensor_near_equal(expected_result, cross_entropy(outputs, targets))


def test_cross_entropy_derivative_basic():
    outputs = np.array([0.5, 0.9, 0.1, 0.0001])
    targets = np.array([0.5, 0.99, 0.01, 0.2])
    expected_result = ((outputs - targets) / (outputs * (1 - outputs)))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(outputs, targets))


def test_cross_entropy_derivative_NumpyTensor():
    outputs = NumpyTensor([0.5, 0.9, 0.1, 0.0001])
    targets = NumpyTensor([0.5, 0.99, 0.01, 0.2])
    expected_result = ((outputs - targets) / (outputs * (1 - outputs)))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(outputs, targets))


@attr('cuda')
def test_cross_entropy_derivative_CudamatTensor():
    from mylearn.backends._cudamat import CudamatTensor
    outputs = CudamatTensor([0.5, 0.9, 0.1, 0.0001])
    targets = CudamatTensor([0.5, 0.99, 0.01, 0.2])
    expected_result = ((outputs - targets) / (outputs * (1 - outputs)))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(outputs, targets))
