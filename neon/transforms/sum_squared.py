# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Sum of squares transform functions and classes.
"""

from neon.transforms.cost import Cost


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
    backend.subtract(outputs, targets, temp[0])
    backend.multiply(temp[0], temp[0], temp[0])
    return 0.5 * backend.sum(temp[0])


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

    backend.subtract(outputs, targets, temp[0])
    return temp[0]


class SumSquaredDiffs(Cost):
    """
    Embodiment of a sum of squared differences cost function.
    """
    def __init__(self, **kwargs):
        super(SumSquaredDiffs, self).__init__(**kwargs)

        # Allocate temporary storage
        # both temp bufs are just nout x batchsize
        tempbuf = self.backend.empty(self.inputbuf1.shape, self.temp_dtype)
        self.temp = [tempbuf]

    def apply_function(self, targets):
        """
        Apply the sum of squared differences cost function to the datasets
        passed.
        """
        return sum_squared_diffs(self.backend, self.inputbuf1,
                                 targets, self.temp)

    def apply_derivative(self, targets):
        """
        Apply the derivative of the sum of squared differences cost function
        to the datasets passed.
        """
        return sum_squared_diffs_derivative(self.backend,
                                            self.inputbuf1, targets,
                                            self.temp)
