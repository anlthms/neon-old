"""
Simple linear transform functions and classes.
"""

import numpy as np

from mylearn.backends._cudamat import CudamatTensor
from mylearn.backends._numpy import NumpyTensor
from mylearn.transforms.activation import Activation


def identity(dataset):
    """
    Applies identity (i.e. no) transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: the dataset.
    """
    return dataset


def identity_derivative(dataset):
    """
    Applies derivative of the identity linear transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    if isinstance(dataset, (int, float)):
        return 1.0
    else:
        res = np.ones(dataset.shape)
        if isinstance(dataset, CudamatTensor):
            return CudamatTensor(res)
        elif isinstance(dataset, NumpyTensor):
            return NumpyTensor(res)
        else:
            return res


class Identity(Activation):
    """
    Embodiment of an identity linear activation function.
    """

    @staticmethod
    def apply_function(dataset):
        """
        Apply the identity linear activation function.
        """
        return identity(dataset)

    @staticmethod
    def apply_derivative(dataset):
        """
        Apply the identity linear activation function derivative.
        """
        return identity_derivative(dataset)
