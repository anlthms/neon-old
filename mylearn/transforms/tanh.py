"""
Hyperbolic tangent transform functions and classes.
"""

import numpy as np

from mylearn.backends._cudamat import Cudamat, CudamatTensor
from mylearn.backends._numpy import Numpy, NumpyTensor
from mylearn.transforms.activation import Activation


def tanh(dataset):
    """
    Applies the hyperbolic tangent transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    exp_fn = np.exp
    if isinstance(dataset, CudamatTensor):
        exp_fn = Cudamat.exp
    elif isinstance(dataset, NumpyTensor):
        exp_fn = Numpy.exp
    res = exp_fn(-2 * dataset)
    return (1.0 - res) / (1.0 + res)


def tanh_derivative(dataset):
    """
    Applies derivative of the hyperbolic tangent transform to the dataset
    passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    res = tanh(dataset)
    return 1.0 - res * res


class Tanh(Activation):
    """
    Embodiment of a hyperbolic tangent activation function.
    """

    @staticmethod
    def apply_function(dataset):
        """
        Apply the hyperbolic tangent activation function.
        """
        return tanh(dataset)

    @staticmethod
    def apply_derivative(dataset):
        """
        Apply the hyperbolic tangent activation function derivative.
        """
        return tanh_derivative(dataset)
