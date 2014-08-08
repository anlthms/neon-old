"""
Cross entropy transform functions and classes.
"""

import numpy

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
    # numpy.ndarray doesn't define log() hence the work-around
    if isinstance(outputs, (int, float, numpy.ndarray)):
        ln_outputs = numpy.log(outputs)
        ln_one_minus_outputs = numpy.log(1 - outputs)
    else:
        ln_outputs = outputs.log()
        ln_one_minus_outputs = (1 - outputs).log()
    return (-targets * ln_outputs -
            (1 - targets) * ln_one_minus_outputs).mean()


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
