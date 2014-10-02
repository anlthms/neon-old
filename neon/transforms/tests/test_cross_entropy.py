from nose.plugins.attrib import attr
import numpy as np

from neon.backends._numpy import Numpy, NumpyTensor
from neon.transforms.cross_entropy import (cross_entropy,
                                              cross_entropy_derivative)
from neon.util.testing import assert_tensor_near_equal


def test_cross_entropy_numpytensor():
    outputs = NumpyTensor([0.5, 0.9, 0.1, 0.0001])
    targets = NumpyTensor([0.5, 0.99, 0.01, 0.2])
    temp = [Numpy.zeros(outputs.shape), Numpy.zeros(outputs.shape)]
    expected_result = np.mean((- targets.raw()) * np.log(outputs.raw()) -
                              (1 - targets.raw()) * np.log(1 - outputs.raw()))
    assert_tensor_near_equal(expected_result, cross_entropy(Numpy, outputs,
                                                            targets, temp))


@attr('cuda')
def test_cross_entropy_cudamattensor():
    from neon.backends._cudamat import Cudamat, CudamatTensor
    c = Cudamat(rng_seed=0)  # to ensure cublas_init() is called.
    outputs = CudamatTensor([0.5, 0.9, 0.1, 0.0001])
    targets = CudamatTensor([0.5, 0.99, 0.01, 0.2])
    temp = [c.zeros(outputs.shape), c.zeros(outputs.shape)]
    expected_result = np.mean((- targets.raw()) * np.log(outputs.raw()) -
                              (1 - targets.raw()) * np.log(1 - outputs.raw()))
    assert_tensor_near_equal(expected_result, cross_entropy(c, outputs,
                                                            targets, temp))


def test_cross_entropy_derivative_numpytensor():
    outputs = NumpyTensor([0.5, 0.9, 0.1, 0.0001])
    targets = NumpyTensor([0.5, 0.99, 0.01, 0.2])
    temp = [Numpy.zeros(outputs.shape), Numpy.zeros(outputs.shape)]
    expected_result = ((outputs.raw() - targets.raw()) /
                       (outputs.raw() * (1 - outputs.raw())))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(Numpy, outputs,
                                                      targets, temp))


@attr('cuda')
def test_cross_entropy_derivative_cudamattensor():
    from neon.backends._cudamat import Cudamat, CudamatTensor
    c = Cudamat(rng_seed=0)
    outputs = CudamatTensor([0.5, 0.9, 0.1, 0.0001])
    targets = CudamatTensor([0.5, 0.99, 0.01, 0.2])
    temp = [c.zeros(outputs.shape), c.zeros(outputs.shape)]
    expected_result = ((outputs.raw() - targets.raw()) /
                       (outputs.raw() * (1 - outputs.raw())))
    assert_tensor_near_equal(expected_result,
                             cross_entropy_derivative(c, outputs,
                                                      targets, temp))
