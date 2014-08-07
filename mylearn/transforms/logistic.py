"""
Logistic transform functions and classes.
"""

from math import exp, log
import numpy as np

from mylearn.backends._cudamat import Cudamat, CudamatTensor
from mylearn.backends._numpy import Numpy, NumpyTensor
from mylearn.transforms.activation import Activation


def logistic(dataset):
    """
    Applies logistic transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    exp_fn = exp
    if isinstance(dataset, CudamatTensor):
        # cudamat has sigmoid function that should probably be called directly
        # to improve speed.
        exp_fn = Cudamat.exp
    elif isinstance(dataset, NumpyTensor):
        exp_fn = Numpy.exp
    elif isinstance(dataset, np.ndarray):
        exp_fn = np.exp
    return (1.0 / (1.0 + exp_fn(- dataset)))


def logistic_derivative(dataset):
    """
    Applies derivative of the logistic transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    return logistic(dataset) * (1 - logistic(dataset))


def pseudo_logistic(dataset):
    """
    Applies faster, approximate logistic transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    return (1.0 / (1.0 + 2 ** (- dataset)))


def pseudo_logistic_derivative(dataset):
    """
    Applies derivative of the approximate logistic transform to the dataset
    passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    res = pseudo_logistic(dataset)
    return (log(2) * res * (1.0 - res))


class Logistic(Activation):
    """
    Embodiment of a logistic activation function.
    """

    @staticmethod
    def apply_function(dataset):
        """
        Apply the logistic activation function.
        """
        return logistic(dataset)

    @staticmethod
    def apply_derivative(dataset):
        """
        Apply the logistic activation function derivative.
        """
        return logistic_derivative(dataset)


class PseudoLogistic(Activation):
    """
    Embodiment of an approximate logistic activation function.
    """

    @staticmethod
    def apply_function(dataset):
        """
        Apply the approximate logistic activation function.
        """
        return pseudo_logistic(dataset)

    @staticmethod
    def apply_derivative(dataset):
        """
        Apply the approximate logistic activation function derivative.
        """
        return pseudo_logistic_derivative(dataset)
