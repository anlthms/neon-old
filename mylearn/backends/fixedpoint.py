"""
Custom dtype based fixed point CPU backend and Tensor class.  Has configurable
integer and fraction bit width, rounding and overflow schemes.
"""

import logging
import numpy as np

from mylearn.backends import fixpt
from mylearn.backends._numpy import Numpy, NumpyTensor

logger = logging.getLogger(__name__)


class FixedPoint(Numpy):
    """
    Sets up a CPU based fixed point backend for matrix ops.
    """

    @staticmethod
    def zeros(shape, dtype=fixpt):
        return FixedPointTensor(np.zeros(shape, dtype))

# temporary helper to ensure elements converted to fixpt
to_fixpt = np.vectorize(lambda x: fixpt(x), otypes=[fixpt, ])

class FixedPointTensor(NumpyTensor):
    """
    CPU based configurable fixed point data structure.

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        dtype (numpy.ndtype, optional): underlying data type of the elements.
                                        Should be a `fixpt` type, parameterized
                                        with an appropriate bit width.  If None
                                        we will use the default `fixpt`
                                        parameters.
    """

    def __init__(self, obj, dtype=fixpt):
        # TODO: understand why obj elements are not being casted to fixpt dtype
        # x = np.array([1, 2, 3, 4], dtype=fixpt)
        # does nothing!  nor does y = x.astype(fixpt)
        super(FixedPointTensor, self).__init__(obj, dtype)
        self._tensor = to_fixpt(self._tensor)
