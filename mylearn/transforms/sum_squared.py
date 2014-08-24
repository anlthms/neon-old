"""
Sum of squares transform functions and classes.
"""

import numpy as np

from mylearn.backends._cudamat import Cudamat, CudamatTensor
from mylearn.backends._numpy import Numpy, NumpyTensor
from mylearn.transforms.cost import Cost


def sum_squared_diffs(backend, outputs, targets, temp):
    """
    Evaluates sum of squared difference on pairwise elements from outputs and
    targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        scalar: Calculated sum of squared diff values for each element.
    """
    backend.subtract(outputs, targets, temp)
    backend.multiply(temp, temp, temp)
    return 0.5 * backend.sum(temp)


def sum_squared_diffs_derivative(backend, outputs, targets, temp):
    """
    Applies derivative of the sum of squared differences to pairwise elements
    from outputs and targets (with respect to the outputs).

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated diff values for each corresponding element.
                    Will have the same shape as outputs.
    """

    backend.subtract(outputs, targets, temp)
    return temp


class SumSquaredDiffs(Cost):
    """
    Embodiment of a sum of squared differences cost function.
    """

    @staticmethod
    def apply_function(backend, outputs, targets, temp):
        """
        Apply the sum of squared differences cost function to the datasets
        passed.
        """
        return sum_squared_diffs(backend, outputs, targets, temp)

    @staticmethod
    def apply_derivative(backend, outputs, targets, temp):
        """
        Apply the derivative of the sum of squared differences cost function
        to the datasets passed.
        """
        return sum_squared_diffs_derivative(backend, outputs, targets, temp)
