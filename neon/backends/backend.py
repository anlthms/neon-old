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

    @classmethod
    def array(cls, obj, dtype=None):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        elements based on obj values.

        Arguments:
            obj (array_like): input array object to construct from.  Can be
                              built-in python scalar or list (of lists), or a
                              numpy.ndarray
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
        Returns:
            Tensor: array object

        Raises:
            NotImplmentedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.zeros`,
            :py:func:`~neon.backends.backend.Backend.ones`
        """
        raise NotImplementedError()

    @classmethod
    def zeros(cls, shape, dtype=None):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        each element with a value of 0.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
        Returns:
            Tensor: array object

        Raises:
            NotImplmentedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.ones`,
            :py:func:`~neon.backends.backend.Backend.array`
        """
        raise NotImplementedError()

    @classmethod
    def ones(cls, shape, dtype=None):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        each element with a value of 1.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
        Returns:
            Tensor: array object

        Raises:
            NotImplmentedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.zeros`,
            :py:func:`~neon.backends.backend.Backend.array`
        """
        raise NotImplementedError()

    @classmethod
    def copy(cls, tsr):
        """
        Construct and return a deep copy of the Tensor passed.

        Arguments:
            tsr (Tensor): the object to copy

        Returns:
            Tensor: new array object with the same values as tsr.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def add(cls, left, right, out):
        """
        Perform element-wise addition on the Tensor operands, storing the
        resultant values in the out Tensor.  Each operand and out must have
        identical shape or be broadcastable as such.

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def subtract(cls, left, right, out):
        """
        Perform element-wise subtraction on the Tensor operands, storing the
        resultant values in the out Tensor.  Each operand and out must have
        identical shape or be broadcastable as such.

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def multiply(cls, left, right, out):
        """
        Perform element-wise multiplication on the Tensor operands, storing the
        resultant values in the out Tensor.  Each operand and out must have
        identical shape or be broadcastable as such.

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def divide(cls, left, right, out):
        """
        Perform element-wise division on the Tensor operands, storing the
        resultant values in the out Tensor.  Each operand and out must have
        identical shape or be broadcastable as such.

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def log(cls, tsr, out):
        """
        Perform element-wise natural logarithm transformation on Tensor tsr,
        storing the result in Tensor out.  Both Tensor's should have identical
        shape.

        Arguments:
            tsr (Tensor): input to be transformed.
            out (Tensor): where the result will be stored.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def exp(cls, tsr, out):
        """
        Perform element-wise exponential transformation on Tensor tsr,
        storing the result in Tensor out.  Both Tensor's should have identical
        shape.

        Arguments:
            tsr (Tensor): input to be transformed.
            out (Tensor): where the result will be stored.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def dot(cls, left, right, out):
        """
        Perform sum product between the last axis of left and the second last
        axis of right, storing the result in out.  Note that this dot product
        is equivalent to the inner product if operands are vectors, and matrix
        multiplication if both operands are matrices.  All Tensor's should have
        the same shape or be broadcastable as such.

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Raises:
            NotImplementedError: Can't be instantiated directly.
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
        object (numpy.ndarray): An n-dimensional array containing the actual
                                data values.  Python scalars, and lists (of
                                lists) are also supported in addition to
                                :py:class:`numpy.ndarray` objects
        dtype (numpy.dtype, optional): The underlying type of each element

    Attributes:
        shape (list): array specifying the length of each dimension
        dtype (numpy.dtype): the underlying type given to each element.

    Raises:
        NotImplmentedError: Can't be instantiated directly.
    """
    shape = None
    dtype = None

    def __init__(self, object, dtype=None):
        raise NotImplementedError()

    def reshape(self, shape):
        """
        Adjusts the dimensions of the data to the specified shape.  The number
        of elements represented by the new shape must be the same as before.

        Arguments:
            shape (int, list): new length of each dimension

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def transpose(self):
        """
        Returns a view of the data in this Tensor whereby rows and column
        elements are swapped.

        Return:
            Tensor: transposed data view.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def take(self, indices, axis=None, out=None):
        """
        Extract a subset of elements along a given axis.

        Arguments:
            indices (list): 0 based element offsets to extract.
            axis (int, optional): The axis over which to select values.  If
                                  None, we index over a 1-D flattened version
                                  of the Tensor
            out (Tensor, optional): Pre-allocated Tensor of the correct size
                                    in which we will write the results.

        Returns:
            Tensor: subset of original elements

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
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
