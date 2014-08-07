"""
Cross entropy transform functions and classes.
"""

import numpy as np

from mylearn.backends._cudamat import Cudamat, CudamatTensor
from mylearn.backends._numpy import Numpy, NumpyTensor
from mylearn.transforms.cost import Cost


def cross_entropy(outputs, targets):
    """
    Evaluates cross entropy on pairwise elements from outputs and targets.

    Arguments:
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and type as outputs.
    """
    # need mean and log
    mean_fn = np.mean
    log_fn = np.log
    if isinstance(outputs, CudamatTensor):
        mean_fn = Cudamat.mean
        log_fn = Cudamat.log
    elif isinstance(outputs, NumpyTensor):
        mean_fn = Numpy.mean
        log_fn = Numpy.log
    return mean_fn(-targets * log_fn(outputs) -
                   (1 - targets) * log_fn(1 - outputs))


def cross_entropy_derivative(outputs, targets):
    """
    Applies derivative of the cross entropy to the pairwise elements from
    outputs and targets.

    Arguments:
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and backend as outputs.
    """
    # negative elements should be set to 0, positive to 1
    return ((outputs - targets) / (outputs * (1.0 - outputs)))


class CrossEntropy(Cost):
    """
    Embodiment of a cross entropy cost function.
    """

    @staticmethod
    def apply_function(outputs, targets):
        """
        Apply the cross entropy cost function to the datasets passed.
        """
        return cross_entropy(outputs, targets)

    @staticmethod
    def apply_derivative(outputs, targets):
        """
        Apply the derivative of the cross entropy cost function to the datasets
        passed.
        """
        return cross_entropy_derivative(outputs, targets)
