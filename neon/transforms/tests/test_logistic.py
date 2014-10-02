from nose.plugins.attrib import attr
import numpy as np

from neon.backends._numpy import Numpy, NumpyTensor
from neon.transforms.logistic import logistic, logistic_derivative
from neon.util.testing import assert_tensor_near_equal


def test_logistic_numpytensor():
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    temp = Numpy.zeros((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    logistic(Numpy, NumpyTensor(inputs), temp)
    assert_tensor_near_equal(NumpyTensor(outputs), temp)


@attr('cuda')
def test_logistic_cudamattensor():
    from neon.backends._cudamat import Cudamat, CudamatTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    c = Cudamat(rng_seed=0)
    temp = c.zeros((3, 1))
    logistic(c, CudamatTensor(inputs), temp)
    assert_tensor_near_equal(CudamatTensor(outputs), temp)


def test_logistic_derivative_numpytensor():
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    outputs = outputs * (1.0 - outputs)
    temp = Numpy.zeros(inputs.shape)
    logistic_derivative(Numpy, NumpyTensor(inputs), temp)
    assert_tensor_near_equal(NumpyTensor(outputs), temp)


@attr('cuda')
def test_logistic_derivative_cudamattensor():
    from neon.backends._cudamat import Cudamat, CudamatTensor
    inputs = np.array([0, 1, -2]).reshape((3, 1))
    outputs = 1.0 / (1.0 + np.exp(-inputs))
    outputs = outputs * (1.0 - outputs)
    c = Cudamat(rng_seed=0)
    temp = c.zeros(inputs.shape)
    logistic_derivative(c, CudamatTensor(inputs), temp)
    assert_tensor_near_equal(CudamatTensor(outputs), temp)
