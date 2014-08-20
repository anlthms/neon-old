"""
Cross entropy transform functions and classes.
"""

from mylearn.transforms.cost import Cost


def cross_entropy(backend, outputs, targets, temp):
    """
    Evaluates cross entropy on pairwise elements from outputs and targets.

    Arguments:
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and type as outputs.
    """
    # Compute (t-1)*log(1-y).
    backend.add(targets, backend.wrap(-1.0), out=temp[0])
    backend.subtract(backend.wrap(1.0), outputs, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(temp[0], temp[1], out=temp[0])

    # Compute t*log(y).
    backend.log(outputs, out=temp[1])
    backend.multiply(targets, temp[1], out=temp[1])

    backend.subtract(temp[0], temp[1], out=temp[0])
    return backend.mean(temp[0])


def cross_entropy_derivative(backend, outputs, targets, temp):
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
    backend.subtract(outputs, targets, out=temp[0])
    backend.subtract(backend.wrap(1.0), outputs, out=temp[1])
    backend.multiply(temp[1], outputs, out=temp[1])
    backend.divide(temp[0], temp[1], out=temp[0])
    return temp[0]


class CrossEntropy(Cost):
    """
    Embodiment of a cross entropy cost function.
    """

    @staticmethod
    def apply_function(backend, outputs, targets, temp):
        """
        Apply the cross entropy cost function to the datasets passed.
        """
        return cross_entropy(backend, outputs, targets, temp)

    @staticmethod
    def apply_derivative(backend, outputs, targets, temp):
        """
        Apply the derivative of the cross entropy cost function to the datasets
        passed.
        """
        return cross_entropy_derivative(backend, outputs, targets, temp)
