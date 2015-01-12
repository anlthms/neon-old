# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Cross entropy transform functions and classes.
"""

from neon.transforms.cost import Cost
from neon.transforms.logistic import Logistic
from neon.transforms.softmax import Softmax
from neon.util.param import opt_param


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
    backend.add(targets, -1.0, out=temp[0])
    backend.subtract(1.0, outputs, out=temp[1])
    backend.clip(temp[1], backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(temp[0], temp[1], out=temp[0])

    # Compute t*log(y).
    backend.clip(outputs, backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(targets, temp[1], out=temp[1])

    # Compute t*log(y) - (t-1)*log(1-y)
    backend.subtract(temp[0], temp[1], out=temp[0])
    return backend.sum(temp[0])


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
    backend.clip(outputs, backend.epsilon, 1, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(targets, temp[1], out=temp[1])
    backend.multiply(temp[1], -1.0, out=temp[0])
    return backend.sum(temp[0])


def cross_entropy_derivative(backend, outputs, targets, temp, scale=1.0):
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
    backend.subtract(1.0, outputs, out=temp[1])
    backend.multiply(temp[1], outputs, out=temp[1])
    backend.clip(temp[1], backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.divide(temp[0], temp[1], out=temp[0])
    return temp[0]


def cross_entropy_multi_derivative(backend, outputs, targets, temp, scale=1.0):
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
    backend.multiply(temp[0], -scale, out=temp[0])
    return temp[0]


def shortcut_derivative(backend, outputs, targets, temp, scale=1.0):
    """
    For use when combining cost with matched activation
    i.e. cross_entropy_binary with logistic or
         cross_entropy_multi  with softmax
    Derivative has simpler form and removes numerical errors
    """
    backend.subtract(outputs, targets, out=temp[0])
    backend.multiply(temp[0], scale, out=temp[0])
    return temp[0]


class CrossEntropy(Cost):

    """
    Embodiment of a cross entropy cost function.
    """
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)

    def initialize(self, kwargs):
        opt_param(self, ['shortcut_deriv'], True)
        super(CrossEntropy, self).initialize(kwargs)
        if isinstance(self.olayer.activation, Softmax):
            self.ce_function = cross_entropy_multi
            if self.shortcut_deriv:
                self.cd_function = shortcut_derivative
                self.olayer.skip_act = True
            else:
                self.cd_function = cross_entropy_multi_derivative
        elif isinstance(self.olayer.activation, Logistic):
            self.ce_function = cross_entropy
            if self.shortcut_deriv:
                self.cd_function = shortcut_derivative
                self.olayer.skip_act = True
            else:
                self.cd_function = cross_entropy_derivative
        else:
            self.ce_function = cross_entropy
            self.cd_function = cross_entropy_derivative

    def __str__(self):
        return ("Cost Function: {bnry} {shrtct}\n".format(
                bnry=self.use_binary, shrtct=self.shortcut_deriv))

    def set_outputbuf(self, databuf):
        if not self.outputbuf or self.outputbuf.shape != databuf.shape:
            tempbuf = self.backend.empty(databuf.shape, self.temp_dtype)
            self.temp = [tempbuf, tempbuf.copy()]
        self.outputbuf = databuf

    def set_berrbuf(self):
        # used by layer2 only.
        return self.temp[0]

    def apply_function(self, targets):
        """
        Apply the cross entropy cost function to the datasets passed.
        """
        return self.ce_function(self.backend, self.outputbuf,
                                targets, self.temp) * self.scale

    def apply_derivative(self, targets):
        """
        Apply the derivative of the cross entropy cost function to the datasets
        passed.
        """
        return self.cd_function(self.backend, self.outputbuf,
                                targets, self.temp, self.scale)
