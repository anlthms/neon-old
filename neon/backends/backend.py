# pylint: disable = R0904, R0913, C0103
"""
Houses low-level code for performing underlying data manipulation operations.
"""

from neon.util.persist import YAMLable


class Backend(YAMLable):
    """
    Generic backend used to manipulate data.  This abstract
    base class defines what operation each concrete backend must support.
    Inherits configuration file handling via `yaml.YAMLObject
    <http://pyyaml.org/wiki/PyYAMLDocumentation#YAMLObject>`_

    Notes:
        See the list of `implemented backends </backends.html>`_
    """

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

    def rng_init(self):
        """
        Perform random number initialization.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError("Can't create direct instances of Backend")

    def uniform(self, low=0.0, high=1.0, size=1):
        """
        Uniform random number generation of samples in range [low, high).

        Arguments:
            low (float, optional): Minimal sample value.  Defaults to 0.0
            high (float, optional): Maximal sample value (open-ended range).
                                    Defaults to 1.0.
            size (int, optional): The number of samples to return.  Defaults
                                  to 1

        Returns:
            Tensor: of size size filled with these random numbers.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError("Can't create direct instances of Backend")

    def normal(self, loc=0.0, scale=1.0, size=1):
        """
        Gaussian/Normal random number generation of samples centered around
        mean loc, and with standard deviation scale.

        Arguments:
            loc (float, optional): Central value for Gaussian.  Defaults to 0.0
            scale (float, optional): Standard deviation for samples.  Defaults
                                     to 1.0
            size (int, optional): The number of samples to return.  Defaults
                                  to 1

        Returns:
            Tensor: of size size filled with these random numbers.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError("Can't create direct instances of Backend")


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
            Tensor: log transformed values

        Raises:
            NotImplmentedError: Must override in a child Tensor class
        """
        raise NotImplementedError()

    def exp(self):
        """
        Exponentiates each element of this tensor.

        Returns:
            Tensor: e raised to the power of each value

        Raises:
            NotImplmentedError: Must override in a child Tensor class
        """
        raise NotImplementedError()
