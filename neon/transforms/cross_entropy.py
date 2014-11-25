# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Cross entropy transform functions and classes.
"""

from neon.transforms.cost import Cost


def cross_entropy(backend, outputs, targets, temp):
    """
    Evaluates cross entropy on pairwise elements from outputs and targets.

    Given that this is undefined for predicted outputs equal to exactly 0 or
    1.0, we first clip these outputs to epsilon (backend machine precision) and
    1.0 - epsilon respectively.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and type as outputs.
    """
    # Compute (t-1)*log(1-y).
    backend.add(targets, backend.wrap(-1.0), out=temp[0])
    backend.subtract(backend.wrap(1.0), outputs, out=temp[1])
    backend.clip(temp[1], backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(temp[0], temp[1], out=temp[0])

    # Compute t*log(y).
    backend.clip(outputs, backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(targets, temp[1], out=temp[1])

    backend.subtract(temp[0], temp[1], out=temp[0])
    return backend.mean(temp[0])


def cross_entropy_multi(backend, outputs, targets, temp):
    """
    Evaluates cross entropy on elements from outputs and targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and type as outputs.
    """

    # Compute (t*log(y)).
    backend.log(outputs, out=temp[1])
    backend.multiply(targets, temp[1], out=temp[1])
    backend.multiply(temp[1], backend.wrap(-1.0), out=temp[0])
    return backend.mean(temp[0])


def cross_entropy_derivative(backend, outputs, targets, temp):
    """
    Applies derivative of the cross entropy to the pairwise elements from
    outputs and targets.

    Note that this is undefined for predicted outputs equal to exactly 0 or
    1.0, so we clip these to epsilon (backend machine precision) and 1.0 -
    epsilon respectively.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and backend as outputs.
    """
    backend.subtract(outputs, targets, out=temp[0])
    backend.subtract(backend.wrap(1.0), outputs, out=temp[1])
    backend.multiply(temp[1], outputs, out=temp[1])
    backend.clip(temp[1], backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.divide(temp[0], temp[1], out=temp[0])
    return temp[0]


def cross_entropy_multi_derivative(backend, outputs, targets, temp):
    """
    Applies derivative of the cross entropy to the pairwise elements from
    outputs and targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and backend as outputs.
    """
    backend.divide(targets, outputs, out=temp[0])
    backend.multiply(temp[0], backend.wrap(-1.0), out=temp[0])
    return temp[0]


def shortcut_derivative(backend, outputs, targets, temp):
    """
    For use when combining cost with matched activation
    i.e. cross_entropy_binary with logistic or
         cross_entropy_multi  with softmax
    Derivative has simpler form and removes numerical errors
    """
    backend.subtract(targets, outputs, out=temp[0])
    return temp[0]


class CrossEntropy(Cost):

    """
    Embodiment of a cross entropy cost function.
    """
    def __init__(self, use_binary=True, shortcut_deriv=False):
        self.useBinary = use_binary
        self.shortcutDeriv = shortcut_deriv

    def apply_function(self, backend, outputs, targets, temp):
        """
        Apply the cross entropy cost function to the datasets passed.
        """
        if self.useBinary:
            ce_function = cross_entropy
        else:
            ce_function = cross_entropy_multi

        return ce_function(backend, outputs, targets, temp)

    def apply_derivative(self, backend, outputs, targets, temp):
        """
        Apply the derivative of the cross entropy cost function to the datasets
        passed.
        """
        if self.shortcutDeriv:
            cd_function = shortcut_derivative
        else:
            if self.useBinary:
                cd_function = cross_entropy_derivative
            else:
                cd_function = cross_entropy_multi_derivative

        return cd_function(backend, outputs, targets, temp)
