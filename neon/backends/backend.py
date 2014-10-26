# pylint: disable = R0904, R0913, C0103
"""
Houses low-level code for performing underlying data manipulation operations.
"""

from neon.util.persist import YAMLable


class Backend(YAMLable):
    """
    backend interface used to manipulate Tensor data.  This abstract
    base class defines what operations each concrete backend must support.
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

    def uniform(self, low=0.0, high=1.0, size=1):
        """
        Uniform random number generation of samples in range [low, high).

        Arguments:
            low (float, optional): Minimal sample value.  Defaults to 0.0
            high (float, optional): Maximal sample value (open-ended range).
                                    Defaults to 1.0.
            size (int, list, optional): The shape of the samples to return.
                                        Defaults to 1

        Returns:
            Tensor: of shape size filled with these random numbers.

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.normal`,
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
            size (int, list, optional): The shape of the samples to return.
                                        Defaults to 1

        Returns:
            Tensor: of shape size filled with these random numbers.

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.uniform`,
        """
        raise NotImplementedError("Can't create direct instances of Backend")

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

        Returns:
            Tensor: reference to out

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

        Returns:
            Tensor: reference to out

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

        Returns:
            Tensor: reference to out

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

        Returns:
            Tensor: reference to out

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

        Returns:
            Tensor: reference to out

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

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def power(self, tsr, power, out):
        """
        Perform element-wise raise of tsr values to specified power,
        storing the result in Tensor out.  Both Tensor's should have identical
        shape.

        Arguments:
            tsr (Tensor): input to be transformed.
            power (numeric): exponentiated value to be applied to element.
                             Examples include 2 (square), 0.5 (sqaure root).
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

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

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def not_equal(self, left, right, out):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def greater(self, left, right, out):
        """
        Performs element-wise greater than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def greater_equal(self, left, right, out):
        """
        Performs element-wise greater than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def less(self, left, right, out):
        """
        Performs element-wise less than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def less_equal(self, left, right, out):
        """
        Performs element-wise less than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def sum(cls, tsr, axes, out):
        """
        Calculates the summation of the elements along the specified axes.

        Arguments:
            tsr (Tensor): the Tensor on which to perform the sum
            axes (int, list, optional): the dimension(s) along which to sum.
                                        If set to None, we will sum over all
                                        dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def mean(cls, tsr, axes, out):
        """
        Calculates the arithmetic mean of the elements along the specified
        axes.

        Arguments:
            tsr (Tensor): the Tensor on which to compute the average
            axes (int, list, optional): the dimension(s) along which to
                                        average.  If set to None, we will
                                        average over all dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def min(cls, tsr, axis, out):
        """
        Calculates the minimal element value along the specified axis.

        Arguments:
            tsr (Tensor): the Tensor on which to compute the minimum
            axis (int, optional): the dimension along which to find the
                                  minimum.  If set to None, we will
                                  compute the overall minimal value
                                  across all dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def max(cls, tsr, axis, out):
        """
        Calculates the maximal element value along the specified axis.

        Arguments:
            tsr (Tensor): the Tensor on which to compute the maximum
            axis (int, optional): the dimension along which to find the
                                  maximum.  If set to None, we will
                                  compute the overall maximal value
                                  across all dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def argmin(cls, tsr, axis, out):
        """
        Calculates the indices of the minimal element value along the specified
        axis.  If multiple elements contain the minimum, only the indices of
        the first are returned.

        Arguments:
            tsr (Tensor): the Tensor on which to find the minimum indices
            axis (int, optional): the dimension along which to find the
                                  minimum.  If set to None, we will
                                  return the index relative to the 1-D
                                  flattened version of the tensor.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    @classmethod
    def argmax(cls, tsr, axis, out):
        """
        Calculates the indices of the maximal element value along the specified
        axis.  If multiple elements contain the maximum, only the indices of
        the first are returned.

        Arguments:
            tsr (Tensor): the Tensor on which to find the maximum index
            axis (int, optional): the dimension along which to find the
                                  maximum.  If set to None, we will
                                  return the index the relative to the 1-D
                                  flattened version of the tensor.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def norm(self, tsr, order=None, axis=None):
        """
        Calculates and returns the p-norm of the Tensor along the specified
        axis.  The p-norm is defined on A as
        :math:`||A||_p = \sum_i(|A_i|^p)^{1/p}`.

        Arguments:
            tsr (Tensor): the Tensor on which to find the non-zero indices
            order (int, optional): The order or p upon which the norm is
                                   calculated.  Valid values include:
                                   None, inf, -inf, 0, 1, -1, 2, -2, ...
            axis (int, optional): The axis along which to compute the norm.

        Returns:
            Tensor: p-norm of tsr along the specified axis.

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

    def err_init(self):
        """
        Perform error handling initialization.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()


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
        raw (object): the underlying backend specific data structure.  Could be
                      numpy.ndarray, cudamat.CUDAMatrix, etc. depending on the
                      backend.

    Raises:
        NotImplmentedError: Can't be instantiated directly.
    """
    shape = None
    dtype = None
    raw = None

    def __init__(self, object, dtype=None):
        raise NotImplementedError()

    def __getitem__(self, key):
        """
        Extract a subset view of the items via fancy indexing. e.g. A[5:10, :]

        Notes:
            This approach tends to be slower in speed than
            :py:func:`~neon.backends.backend.Tensor.take`, so use of that is
            recommended.

        Arguments:
            key (int, slice): indices of the slice to take

        Returns:
            Tensor: view of self corresponding to the subset items.

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Tensor.take`,
        """

    def __setitem__(self, key, value):
        """
        Assign the specified value to a subset of elements found by fancy
        indexing.

        Notes:
            This approach tends to be slower in speed than
            :py:func:`~neon.backends.backend.Tensor.take`, so use of that is
            recommended.

        Arguments:
            key (int, slice): indices of the slice to be assigned
            value (numeric array, Tensor): values to be assigned to the
                                          extracted element subset.  If an
                                          array it should be the same shape
                                          as what key indexes (or be
                                          broadcastable as such).

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Tensor.take`,
        """

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

    def repeat(self, repeats, axis=None):
        """
        Repeat elements of an array relative to the specified axis.

        Arguments:
            repeats (int, list): The number of repetitions of each element.  It
                                 will be broadcast to fit the shape of the
                                 given axis
            axis (int, optional): The axis along which to repeat values.  If
                                  set to None, we flatten input to 1-D.

        Returns:
            Tensor: new variant with the same dimensions as self except along
                    the specified axis, where it will contain the repeated
                    elements.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """

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

        See Also:
            :py:func:`~neon.backends.backend.Tensor.__getitem__`,
            :py:func:`~neon.backends.backend.Tensor.__setitem__`,
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
