from math import exp

from nose.plugins.attrib import attr
import numpy as np

from mylearn.backends._numpy import NumpyTensor
from mylearn.transforms.logistic import logistic, logistic_derivative
from mylearn.util.testing import assert_tensor_near_equal


def test_logistic_basics():
    assert_tensor_near_equal(np.array([0.5, (1.0 / (1.0 + exp(-1))),
                                      (1.0 / (1.0 + exp(2)))]),
                             logistic(np.array([0, 1, -2])))


def test_logistic_NumpyTensor():
    assert_tensor_near_equal(NumpyTensor([0.5, (1.0 / (1.0 + exp(-1))),
                                         (1.0 / (1.0 + exp(2)))]),
                             logistic(NumpyTensor([0, 1, -2])))


@attr('cuda')
def test_logistic_CudamatTensor():
    from mylearn.backends._cudamat import CudamatTensor
    assert_tensor_near_equal(CudamatTensor([0.5, (1.0 / (1.0 + exp(-1))),
                                           (1.0 / (1.0 + exp(2)))]),
                             logistic(CudamatTensor([0, 1, -2])))


def test_logistic_derivative_basics():
    assert_tensor_near_equal(np.array([logistic(0) * (1 - logistic(0)),
                                       logistic(1) * (1 - logistic(1)),
                                       logistic(-2) * (1 - logistic(-2))]),
                             logistic_derivative(np.array([0, 1, -2])))


def test_logistic_derivative_NumpyTensor():
    assert_tensor_near_equal(NumpyTensor([logistic(0) * (1 - logistic(0)),
                                          logistic(1) * (1 - logistic(1)),
                                          logistic(-2) * (1 - logistic(-2))]),
                             logistic_derivative(NumpyTensor([0, 1, -2])))


@attr('cuda')
def test_logistic_derivative_CudamatTensor():
    from mylearn.backends._cudamat import CudamatTensor
    assert_tensor_near_equal(CudamatTensor([logistic(0) * (1 - logistic(0)),
                                            logistic(1) * (1 - logistic(1)),
                                            logistic(-2) *
                                            (1 - logistic(-2))]),
                             logistic_derivative(CudamatTensor([0, 1, -2])))
