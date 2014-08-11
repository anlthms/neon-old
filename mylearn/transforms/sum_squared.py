"""
Sum of squares transform functions and classes.
"""

import numpy as np

from mylearn.backends._cudamat import Cudamat, CudamatTensor
from mylearn.backends._numpy import Numpy, NumpyTensor
from mylearn.transforms.cost import Cost


def sum_squared_diffs(outputs, targets):
    """
    Evaluates sum of squared difference on pairwise elements from outputs and
    targets.

    Arguments:
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.

    Returns:
        array_like: Calculated sum of squared diff values for each element.
                    Will have the same shape and type as outputs.
    """
    sum_fn = np.sum
    if isinstance(outputs, CudamatTensor):
        sum_fn = Cudamat.sum
    elif isinstance(outputs, NumpyTensor):
        sum_fn = Numpy.sum
    return 0.5 * sum_fn((outputs - targets) ** 2)


def sum_squared_diffs_derivative(outputs, targets):
    """
    Applies derivative of the sum of squared differences to pairwise elements
    from outputs and targets (with respect to the outputs).

    Arguments:
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and backend as outputs.
    """
    return outputs - targets


class SumSquaredDiffs(Cost):
    """
    Embodiment of a sum of squared differences cost function.
    """

    @staticmethod
    def apply_function(outputs, targets):
        """
        Apply the sum of squared differences cost function to the datasets
        passed.
        """
        return sum_squared_diffs(outputs, targets)

    @staticmethod
    def apply_derivative(outputs, targets):
        """
        Apply the derivative of the sum of squared differences cost function
        to the datasets passed.
        """
        return sum_squared_diffs_derivative(outputs, targets)
