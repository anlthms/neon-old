"""
hinge cost functions and classes for balance networks
"""

from neon.transforms.cost import Cost


def hinge_l2(backend, outputs, targets, temp, subidx=None):
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
    # Compute   0.5 * max(0, (1-t*y))^2
    backend.multiply(targets, outputs, out=temp[0])
    backend.subtract(backend.wrap(1.0), temp[0], out=temp[0])
    backend.greater(temp[0], backend.wrap(0), out=temp[0])
    backend.multiply(temp[0], temp[0], out=temp[0])

    if (subidx is not None and subidx < temp[0].shape[temp[0].major_axis()]):
        temp[0].set_major_slice(subidx, temp[0].shape[temp[0].major_axis()],
                                backend.wrap(0.0))

    backend.multiply(temp[0], backend.wrap(0.5), out=temp[0])

    return backend.sum(temp[0])


def hinge_l2_derivative(backend, outputs, targets, temp, subidx=None):
    """
    Applies derivative of the hinge l2 function to the pairwise elements from
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
    # Compute   -y * max(0, (1-t*y))
    backend.multiply(targets, outputs, out=temp[0])
    backend.subtract(backend.wrap(1.0), temp[0], out=temp[0])
    backend.greater(temp[0], backend.wrap(0), out=temp[0])
    if (subidx is not None and subidx < temp[0].shape[temp[0].major_axis()]):
        temp[0].set_major_slice(subidx, temp[0].shape[temp[0].major_axis()],
                                backend.wrap(0.0))
    backend.multiply(temp[0], targets, out=temp[0])

    backend.multiply(temp[0], backend.wrap(-1.0), out=temp[0])

    return temp[0]


class HingeL2(Cost):

    """
    Embodiment of a Hinge L2 cost function.
    """
    def __init__(self, stopidx=None):
        self.stopidx = stopidx

    def apply_function(self, backend, outputs, targets, temp):
        """
        Apply the cross entropy cost function to the datasets passed.
        """
        return hinge_l2(backend, outputs, targets, temp, subidx=self.stopidx)

    def apply_derivative(self, backend, outputs, targets, temp):
        """
        Apply the derivative of the cross entropy cost function to the datasets
        passed.
        """
        return hinge_l2_derivative(backend, outputs, targets, temp,
                                   subidx=self.stopidx)
