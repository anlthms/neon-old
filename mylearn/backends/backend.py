"""
Houses low-level code for performing underlying data manipulation operations.
"""

import logging
import yaml

logger = logging.getLogger(__name__)


class Backend(yaml.YAMLObject):
    """
    Abstract backend defines operations that must be supported.

    Inherits from yaml.YAMLObject, typically you would utilize a concrete
    child of this class.

    Attributes:
        yaml_loader (yaml.SafeLoader): parser used to load backend.
    """
    yaml_loader = yaml.SafeLoader

    @staticmethod
    def array(obj, dtype=None, copy=True, order=None, subok=False,
              ndim=0):
        """
        Instantiate a new instance of this backend's Tensor class.

        Arguments:
            obj (array_like): input array object to construct from.
            dtype (data-type, optional): numpy dtype to specify size of each
                                         element.
            copy (bool, optional): create a copy of the object.
            order ({'C', 'F', 'A'}, optional): C vs Fortran contiguous order
            subok (bool, optional): pass-through sub classes if True.
                                    Otherwise we force the returned
                                    array to the base class array.
            ndim (int, optional): Minimum number of dimensions output array
                                  should have.  Ones are prepended to meet
                                  this requirement.
        Returns:
            Tensor: array object

        Raises:
            NotImplmentedError: Can't be instantiated directly.
        """
        raise NotImplementedError()


class Tensor(object):
    """
    Represents an arbitrary n-dimensional array data structure.

    Arguments:
        object (numpy.ndarray): An in-memory n-dimensional array containing
                                the data values.
        dtype (numpy.dtype, optional): The underlying type of each element

    Raises:
        NotImplmentedError: Can't be instantiated directly.
    """
    def __init__(self, object, dtype=None):
        raise NotImplementedError()

    def log(self):
        """
        Computes the elementwise natural logarithmic transform on this tensor.

        Returns:
            NumpyTensor: log transformed values

        Raises:
            NotImplmentedError: Must override in a child Tensor class
        """
        raise NotImplementedError()

    def exp(self):
        """
        Exponentiates each element of this tensor.

        Returns:
            NumpyTensor: e raised to the power of each value

        Raises:
            NotImplmentedError: Must override in a child Tensor class
        """
        raise NotImplementedError()
