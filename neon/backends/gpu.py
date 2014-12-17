# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Our GPU based backend interface and tensor data structure.  Our implementation
is derived from `cuda-convnet2 <https://code.google.com/p/cuda-convnet2/>`_
"""

import logging
import numpy
from neon.util.compat import MPI_INSTALLED, range
if MPI_INSTALLED:
    from mpi4py import MPI
import math
import cudanet
import os

from neon.backends.backend import Backend, Tensor
from neon.util.error import TooSlowToImplementError

logger = logging.getLogger(__name__)


class GPUTensor(Tensor):

    """
    Our n-dimensional array data structure that can reside on host or on GPU
    device.  Our implementation is a wrapped `cudanet.CUDAMatrix` tensor, where
    cudanet is derived from cuda-convnet2.

    Arguments:
        obj (numpy.ndarray): the actual data values (will be converted
                             to a 2-d row matrix).  Python built-in types like
                             lists and tuples are also supported.
        dtype (None, optional): underlying data type of the elements.
                                Not needed/used for this backend.
    """
    _tensor = None

    def __init__(self, obj, dtype=None):
        if type(obj) == cudanet.CUDAMatrix:
            self._tensor = obj
            self.shape = self._tensor.shape
        else:
            if type(obj) == list:
                obj = numpy.array(obj)
            if type(obj) == numpy.ndarray:
                # CUDAMatrix only supports ndarrays with exactly 2 dimensions
                # (though the elements can be tuples/lists to create arbitrary
                # n dimensions)
                while obj.ndim < 2:
                    obj = obj.reshape(obj.shape + (1, ))
                if obj.ndim != 2:
                    raise ValueError("CUDAMatrix only supports 2-D"
                                     "matrices.  You specifed %d-D" %
                                     obj.ndim)
                logger.debug('Copying to GPU')
                if dtype not in (numpy.float32, numpy.int32) or dtype is None:
                    logger.debug('Dtype %s is unsupported in GPU '
                                 'backend, defaulting float32', dtype)
                    obj = numpy.array(obj, dtype=numpy.float32)
                self._tensor = cudanet.CUDAMatrix(obj)
                self.shape = self._tensor.shape
            else:
                self._tensor = obj
        self.dtype = dtype

    def __str__(self):
        """
        Display a suitable representation of this Tensor.
        Note that this operation requires copying to host.

        Returns:
            str: the representation
        """
        return str(self._tensor.asarray())

    def __getstate__(self):
        """
        Defines what and how we go about serializing an instance of this class.

        Returns:
            numpy.ndarray: Representation of the underlying
                           `cudanet.CUDAMatrix` tensor
        """
        if type(self._tensor) == cudanet.CUDAMatrix:
            return self._tensor.asarray()
        else:
            return self._tensor

    def __setstate__(self, state):
        """
        Defines how we go about deserializing into an instance of this class.

        Arguments:
            state (numpy.ndarray): Serialized representation of the underlying
                                   `cudanet.CUDAMatrix` tensor to be unpacked.
        """
        self.__init__(state)
        if not hasattr(cudanet.CUDAMatrix, 'ones'):
            cudanet.cublas_init()

    def _slice_dim(self, _slice, dim=0):
        """
        Helper that actually performs a slice along the dimension passed.

        Arguments:
            _slice (int or slice): actual slice object specifying indices
            dim (int): dimension number. 0 is for rows, 1 for columns, etc.

        Returns:
            GPUTensor: view or new sliced copy

        Raises:
            TooSlowToImplementError: if invalid `_slice` provided (too
            complex to implement quickly).
        """
        res = self
        fn = res._tensor.row_slice_view
        if dim == 1:
            fn = res._tensor.col_slice_view
        if isinstance(_slice, int):
            _slice = slice(_slice, _slice + 1)
        if isinstance(_slice, slice):
            assert _slice.step is None or _slice.step == 1
            start, stop, stride = _slice.indices(self.shape[dim])
            res = GPUTensor(fn(start, stop))
        elif _slice is Ellipsis:
            pass
        else:
            # arbitrary long list, too expensive to support?
            raise TooSlowToImplementError("column idx too complex")
        return res

    def __getitem__(self, key):
        """
        Extract a subset of elements from this tensor as specified by key.

        Arguments:
            key (tuple, int): the indices to extract/slice along.

        Returns:
            GPUTensor: view or new sliced copy

        Raises:
            IndexError: if invalid number of dimensions specified in key.
        """
        res = self
        if isinstance(key, tuple):
            if len(key) > 2:
                raise IndexError("CUDAMatrix only supports 2-D matrices")
            else:
                for idx in range(len(key) - 1, -1, -1):
                    res = res._slice_dim(key[idx], idx)
        else:
            res = res._slice_dim(key, 0)
        return res

    def __setitem__(self, key, value):
        """
        Assign values to a subset of elements from this tensor.

        Arguments:
            key (tuple, int): The indices to which we assign the values
            value (GPUTensor, int, float): The values to assign at each
                                               key position.  Must be scalar
                                               or if a GPUTensor, must
                                               have the right shape.

        Returns:
            GPUTensor: update view of this tensor

        Raises:
            IndexError: if invalid number of dimensions specified in key.
            NotImplementedError: if invalid value type passed.
            TooSlowToImplementError: if arbitrarily indexed key passed.
        """
        if isinstance(value, GPUTensor):
            value = value._tensor
        elif not isinstance(value, (int, float)):
            raise NotImplementedError("can only assign GPUTensor's or "
                                      "numeric scalars")
        if isinstance(key, tuple):
            if len(key) > 2:
                raise IndexError("CUDAMatrix only supports 2-D matrices")
            elif len(key) == 2:
                if isinstance(key[0], slice):
                    start, stop, stride = key[0].indices(self.shape[0])
                    if start == 0 and stop == self.shape[0]:
                        if isinstance(key[1], slice):
                            start, stop, stride = (key[1].indices(self.
                                                                  shape[1]))
                            self._tensor.set_col_slice(start, stop, value)
                        elif isinstance(key[1], int):
                            self._tensor.set_col_slice(key[1], key[1] + 1,
                                                       value)
                        else:
                            raise TooSlowToImplementError("arbitrary "
                                                          "indexing")
                    elif isinstance(key[1], slice):
                        start_1, stop_1, stride_1 = (key[1].indices(self.
                                                                    shape[1]))
                        if start_1 == 0 and stop_1 == self.shape[1]:
                            self._tensor.set_row_slice(start, stop, value)
                        else:
                            raise TooSlowToImplementError("arbitrary "
                                                          "indexing")
                    else:
                        raise TooSlowToImplementError("arbitrary "
                                                      "indexing")
                elif isinstance(key[0], int):
                    if isinstance(key[1], slice):
                        start_1, stop_1, stride_1 = (key[1].indices(self.
                                                                    shape[1]))
                        if start_1 == 0 and stop_1 == self.shape[1]:
                            self._tensor.set_row_slice(key[0], key[0] + 1,
                                                       value)
                        else:
                            raise TooSlowToImplementError("arbitrary "
                                                          "indexing")
                    else:
                        raise TooSlowToImplementError("arbitrary "
                                                      "indexing")
                else:
                    raise TooSlowToImplementError("arbitrary "
                                                  "indexing")
        else:
            # 1-D index, unless of form x[:] = value, we treat this as
            # x[key, :] = value
            if isinstance(key, slice):
                start, stop, stride = key.indices(self.shape[0])
                if start == 0 and stop == self.shape[0]:
                    # form x[:] = value
                    self._tensor.assign(value)
                else:
                    self._tensor.set_row_slice(start, stop, value)
            else:
                self._tensor.set_row_slice(key, key + 1, value)

    def __delitem__(self, key):
        raise ValueError("cannot delete array elements")

    def asnumpyarray(self):
        self._tensor.copy_to_host()
        return self._tensor.numpy_array

    def __float__(self):
        raise NotImplementedError()

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        """
        Perform element-wise addition with the items in other.
        Now supports limited broadcasting to add a vector to a matrix.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same
                            dimensions as this Tensor, or be broadcastable
                            as such.

        Returns:
            GPUTensor: containing the element-wise sum values.
        """

        if self.shape == other.shape:
            target = cudanet.empty(self.shape)
            if isinstance(other, GPUTensor):
                self._tensor.add(other._tensor, target)
            else:
                self._tensor.add(other, target)
            return GPUTensor(target)
        else:
            if other.shape[1] == 1:  # [Nx1] vector
                ones = cudanet.empty((self.shape[0], 1))
                ones.assign(1)
                # outer product repmat (probably quite inefficient)
                other = GPUTensor(cudanet.dot(ones, other._tensor.transpose()))
            else:  # [1xN] vector
                ones = cudanet.empty((self.shape[0], 1))
                ones.assign(1)
                other = GPUTensor(cudanet.dot(ones, other._tensor))
            target = cudanet.empty(self.shape)
            if isinstance(other, GPUTensor):
                self._tensor.add(other._tensor, target)
            else:
                self._tensor.add(other, target)
            return GPUTensor(target)

    def __radd__(self, other):
        """
        Perform element-wise addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same
                            dimensions as this Tensor, or be broadcastable
                            as such.

        Returns:
            GPUTensor: containing the element-wise sum values.
        """
        target = cudanet.empty(self.shape)
        if isinstance(other, GPUTensor):
            self._tensor.add(other._tensor, target)
        else:
            self._tensor.add(other, target)
        return GPUTensor(target)

    def __iadd__(self, other):
        """
        Perform element-wise in-place addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same
                            dimensions as this Tensor, or be broadcastable
                            as such.

        Returns:
            GPUTensor: updated view of this Tensor
        """
        if isinstance(other, GPUTensor):
            self._tensor.add(other._tensor)
        else:
            self._tensor.add(other)
        return self

    def __sub__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, GPUTensor):
            self._tensor.subtract(other._tensor, target)
        else:
            self._tensor.subtract(other, target)
        return GPUTensor(target)

    def __rsub__(self, other):
        target = cudanet.empty(self.shape)
        self._tensor.mult(-1.0, target)
        if isinstance(other, GPUTensor):
            target.add(other._tensor)
        else:
            target.add(other)
        return GPUTensor(target)

    def __isub__(self, other):
        if isinstance(other, GPUTensor):
            self._tensor.subtract(other._tensor)
        else:
            self._tensor.subtract(other)
        return self

    def __mul__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, GPUTensor):
            self._tensor.mult(other._tensor, target)
        else:
            self._tensor.mult(other, target)
        return GPUTensor(target)

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        if isinstance(other, GPUTensor):
            self._tensor.mult(other._tensor)
        else:
            self._tensor.mult(other)
        return self

    def __div__(self, other):
        # python2 floor rounded division
        return self.__truediv__(other)

    def __truediv__(self, other):
        # python3 fractional division
        target = cudanet.empty(self.shape)
        if isinstance(other, GPUTensor):
            self._tensor.divide(other._tensor, target)
        else:
            self._tensor.divide(other, target)
        return GPUTensor(target)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rtruediv__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, (float, int)):
            other = GPUTensor(other * numpy.ones(self.shape))
        if isinstance(other, GPUTensor):
            other._tensor.divide(self._tensor, target)
        elif isinstance(other, cudanet.CUDAMatrix):
            other.divide(self._tensor, target)
        else:
            return NotImplemented
        return GPUTensor(target)

    def __idiv__(self, other):
        if isinstance(other, GPUTensor):
            self._tensor.divide(other._tensor)
        else:
            self._tensor.divide(other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, GPUTensor):
            self._tensor.divide(other._tensor)
        else:
            self._tensor.divide(other)
        return self

    def __pow__(self, other, modulo=None):
        target = cudanet.empty(self.shape)
        if isinstance(other, GPUTensor):
            cudanet.pow(self._tensor, other._tensor, target)
        else:
            cudanet.pow(self._tensor, other, target)
        return GPUTensor(target)

    def __rpow__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, (float, int)):
            other = GPUTensor(other)
        if isinstance(other, GPUTensor):
            cudanet.pow(other._tensor, self._tensor, target)
        elif isinstance(other, cudanet.CUDAMatrix):
            cudanet.pow(other, self._tensor, target)
        else:
            return NotImplemented
        return GPUTensor(target)

    def __ipow__(self, other):
        if isinstance(other, GPUTensor):
            cudanet.pow(self._tensor, other._tensor)
        else:
            cudanet.pow(self._tensor, other)
        return self

    def copy(self):
        return GPUTensor(self._tensor.copy())

    def raw(self, dohostcopy=True):
        if dohostcopy:
            self._tensor.copy_to_host()
        return self._tensor.numpy_array

    def copy_to_device(self):
        self._tensor.copy_to_device()

    def transpose(self):
        return TransposedGPUTensor(self._tensor, self._tensor.T)

    def reshape(self, shape):
        return GPUTensor(self._tensor.reshape(shape))

    def take(self, indices, axis=None):
        """
        Take returns a subset of a tensor specified by indices.
        Urs modified this to be consistent with numpy, where vectors
        get flipped to always be rows.
        """
        # we only support contiguous indices at the moment because this
        # is all cudanet supports efficiently.
        if isinstance(indices, int):
            indices = [indices, ]  # cudanet only supports 2D matrix
            if self._tensor.shape[0] == 1:
                axis = 1
                # fix the axis if we are dealing with a vector. This is a hack
                # and should be done differently.
        if (indices[-1] - indices[0] == len(indices) - 1):
            if axis == 0:
                return GPUTensor(self._tensor.get_row_slice(indices[0],
                                                            indices[-1] + 1))
            elif axis == 1:
                return GPUTensor(self._tensor.get_col_slice(indices[0],
                                                            indices[-1] + 1))
            elif axis is None:
                # we might be able to do this by first doing a reshape?
                raise TooSlowToImplementError("need to first reshape")
        else:
            raise TooSlowToImplementError("CUDAMatrix can't do arbitrary"
                                          " indexing efficiently")

    def sum(self, axis=None):
        """
        Sum elements of a GPUTensor. If axis is None, all elements are
        summed and a numpy scalar returned. If axis is 1 or 2, sum along that
        axis and return a GPUTensor.
        """
        if axis is None:
            result = self._tensor.sum(axis=0).sum(axis=1)
            logger.debug('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]
        else:
            result = self._tensor.sum(axis=axis)
            logger.debug('major change in functionality of sum')
            return GPUTensor(result)

    def sumsq(self, axis=None):
        """
        Sum of squares of elements of a CudanetTensor. If axis is None,
        all elements are summed and a numpy scalar returned. If axis is 1
        or 2, sum along that axis and return a CudanetTensor.
        """
        if axis is None:
            result = self._tensor.sumsq(axis=None)
            logger.debug('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]
        else:
            result = self._tensor.sumsq(axis=axis)
            logger.debug('major change in functionality of sum')
            return GPUTensor(result)

    def mean(self, axis=None):
        result = self._tensor.mean(axis)
        logger.debug('Copying to host')
        result.copy_to_host()
        return result.numpy_array[0][0]

    def min(self, axis=None):
        result = self._tensor.min(axis)
        logger.debug('Copying to host')
        result.copy_to_host()
        return result.numpy_array[0][0]

    def max(self, axis=None):
        result = self._tensor.max(axis)
        logger.debug('Copying to host')
        result.copy_to_host()
        return result.numpy_array[0][0]

    def log(self):
        target = cudanet.empty(self.shape)
        cudanet.log(self._tensor, target)
        return GPUTensor(target)

    def exp(self):
        target = cudanet.empty(self.shape)
        cudanet.exp(self._tensor, target)
        return GPUTensor(target)


class TransposedGPUTensor(GPUTensor):

    """
    Transposed CUDAMatrix tensor
    """

    def __init__(self, obj, transposed):
        assert type(obj) == cudanet.CUDAMatrix
        self._tensor = transposed
        self.shape = (obj.shape[1], obj.shape[0])


class GPU(Backend):

    """
    Sets up a `cuda-convnet2 <https://code.google.com/p/cuda-convnet2/>`_
    based backend for matrix operations.

    Attributes:
        epsilon (float): the unit roundoff for the elements underlying this
                         tensor.
    """
    # we need to cast epsilon to float to ensure it works with some of the type
    # checking in cudanet functions like less_than() and so forth
    epsilon = float(numpy.finfo(numpy.float32).eps)
    default_dtype = numpy.float32
    tensor_cls = GPUTensor

    def __init__(self, **kwargs):
        # set cuda device to device 0 by default
        cudanet.set_device_id(0)
        self.__dict__.update(kwargs)
        cudanet.cublas_init()
        self.rng_init()

    def __del__(self):
        pass
        # cudanet.cublas_shutdown()
        # the above is what we ought to do, but generates Exceptions due to
        # a known cudanet issue as described here:
        # https://github.com/cudanet/cudanet/issues/19

    def empty(self, shape, dtype=None):
        """
        Instantiate a new instance of the GPUTensor class without initializing
        each element's value.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value (np.float32
                                     unless overridden).

        Returns:
            GPUTensor: newly created data structure reference
        """
        return GPUTensor(cudanet.empty(shape))

    def array(self, obj, dtype=None):
        """
        Instantiate a new instance of the GPUTensor class based on the values
        and shape of obj passed.

        Arguments:
            obj (numpy.ndarray): The n-dimensional array of values to use in
                                 initializing the values of this Tensor.  Note
                                 that python built-in types like scalar
                                 integers and lists are supported.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value (np.float32
                                     unless overridden).

        Returns:
            GPUTensor: newly created data structure reference
        """
        ndarray = numpy.array(obj, dtype=numpy.float32)
        if ndarray.ndim == 1:
            ndarray = ndarray.reshape((1, ndarray.shape[0]))
        return GPUTensor(ndarray)

    def zeros(self, shape, dtype=numpy.float32):
        """
        Instantiate a new instance of the GPUTensor class setting each element
        value to 0.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value (np.float32
                                     unless overridden).

        Returns:
            GPUTensor: newly created data structure reference
        """
        if dtype is None:
            dtype = numpy.float32
        return GPUTensor(cudanet.CUDAMatrix(
            numpy.zeros(shape, dtype=dtype)))

    def ones(self, shape, dtype=numpy.float32):
        """
        Instantiate a new instance of the GPUTensor class setting each element
        value to 1.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value (np.float32
                                     unless overridden).

        Returns:
            GPUTensor: newly created data structure reference
        """
        if dtype is None:
            dtype = numpy.float32
        return GPUTensor(cudanet.CUDAMatrix(
            numpy.ones(shape, dtype=dtype)))

    def wrap(self, obj):
        return GPUTensor(obj)

    def clip(self, a, a_min, a_max, out=None):
        if out is None:
            out = GPUTensor(cudanet.empty((a.shape[0], a.shape[1])))
        cudanet.clip_range(a._tensor, a_min, a_max, out._tensor)
        return out

    def rng_init(self):
        seed = None
        if 'rng_seed' in self.__dict__:
            seed = self.rng_seed
        numpy.random.seed(seed)
        try:
            cudanet.cudanet_init_random(seed)
        except TypeError:
            if seed is not None:
                logger.warn("Must seed random number generator with an "
                            "integer.  You specified: %s" % str(seed))
            cudanet.cudanet_init_random(0)

    def uniform(self, low=0.0, high=1.0, size=1):
        seq = numpy.random.uniform(low, high, size)
        return GPUTensor(numpy.array(seq, dtype=numpy.float32))

    def fill_uniform_thresh(self, a, keepthresh=0.5, dtype=None):
        """
        Uniform random number sample generation.

        Arguments:
            a (dtype): GPUTensor to fill with zeros or ones based on whether
                       sample from uniform distribution is > keepthresh
            keepthresh (float, optional): Minimal sample value that can be
                                          returned. Defaults to 0.5
        Returns:
            Tensor: Of specified size filled with these random numbers.
        """
        # This slow implementation is kept here in commented form should you
        # need to ensure consistency with CPU generated random numbers:
        # a.raw(dohostcopy=False)[:] = numpy.array((numpy.random.uniform(
        #     size=a._tensor.shape) < keepthresh) / keepthresh,
        #     dtype=numpy.float32)
        # a.copy_to_device()

        # This implementation is faster but breaks consistency with CPU
        # backend based random numbers:
        a._tensor.randomize_uniform_thresh(keepthresh=keepthresh)

    def normal(self, loc=0.0, scale=1.0, size=1):
        seq = numpy.random.normal(loc, scale, size)
        return GPUTensor(numpy.array(seq, dtype=numpy.float32))

    def append_bias(self, x):
        """
        Adds a bias row to GPUTensor x, returning a new GPUTensor.
        """
        result = cudanet.empty((x.shape[0] + 1, x.shape[1]))
        result.set_row_slice(0, x.shape[0], x._tensor)
        result.set_row_slice(x.shape[0], (x.shape[0] + 1),
                             cudanet.CUDAMatrix.ones.slice(
                                 0, x.shape[1]).reshape((1, x.shape[1])))
        return GPUTensor(result)

    def copy(self, a):
        assert type(a) == GPUTensor
        return a.copy()

    def dot(self, a, b, out):
        cudanet.dot(a._tensor, b._tensor, out._tensor)

    def add(self, a, b, out):
        if type(a._tensor) != cudanet.CUDAMatrix:
            b._tensor.add(a._tensor, out._tensor)
        else:
            a._tensor.add(b._tensor, out._tensor)

    def subtract(self, a, b, out):
        if type(a._tensor) != cudanet.CUDAMatrix:
            b._tensor.subtract(a._tensor, out._tensor)
            out._tensor.mult(-1.0, out._tensor)
        else:
            a._tensor.subtract(b._tensor, out._tensor)

    def multiply(self, a, b, out):
        a._tensor.mult(b._tensor, target=out._tensor)

    def divide(self, a, b, out):
        a._tensor.divide(b._tensor, out._tensor)

    def reciprocal(self, a, out):
        a._tensor.reciprocal(out._tensor)

    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor): left-hand side operand.
            right (GPUTensor): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        left._tensor.equals(right._tensor, out._tensor)
        return out

    def not_equal(self, left, right, out):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor): left-hand side operand.
            right (GPUTensor): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.equal(left, right, out)
        out._tensor.equals(0, out._tensor)
        return out

    def greater(self, left, right, out):
        """
        Performs element-wise greater than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor): left-hand side operand.
            right (GPUTensor): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        left._tensor.greater_than(right._tensor, out._tensor)
        return out

    def greater_equal(self, left, right, out):
        """
        Performs element-wise greater than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor): left-hand side operand.
            right (GPUTensor): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        # we calculate >= as not <
        left._tensor.less_than(right._tensor, out._tensor)
        out._tensor.equals(0, out._tensor)
        return out

    def less(self, left, right, out):
        """
        Performs element-wise less than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor): left-hand side operand.
            right (GPUTensor): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        left._tensor.less_than(right._tensor, out._tensor)
        return out

    def less_equal(self, left, right, out):
        """
        Performs element-wise less than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor): left-hand side operand.
            right (GPUTensor): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        # we calculate <= as not >
        left._tensor.greater_than(right._tensor, out._tensor)
        out._tensor.equals(0, out._tensor)
        return out

    def norm(self, tsr, order=None, axis=None, out=None):
        """
        Calculates and returns the vector p-norms of the GPUTensor along the
        specified axis.  The p-norm is defined on a vector A as
        :math:`||A||_p = \sum_i(|A_i|^p)^{1/p}`.

        Arguments:
            tsr (GPUTensor): the GPUTensor on which to find the norms
            order (int): The order or p upon which the norm is calculated.
                         Valid values include:
                         None, inf, -inf, 0, 1, -1, 2, -2, ...
            axis (int): The axis along which to compute vector norms.
            out (GPUTensor, optional): where to write the results to.  Must be
                                       of the expected result shape.  If not
                                       specified, a new buffer is created and
                                       returned.

        Returns:
            GPUTensor: p-norm of tsr along the specified axis.
        """
        if not isinstance(axis, int):
            raise AttributeError("invalid axis value: %s", axis)
        if order == float('Inf'):
            res = self.max(self.fabs(tsr), axis)
        elif order == float('-Inf'):
            res = self.min(self.fabs(tsr), axis)
        elif order == 0:
            tmp = self.zeros(tsr.shape)
            self.not_equal(tsr, tmp, tmp)
            res = tmp.sum(axis)
        else:
            res = ((self.fabs(tsr) ** order).sum(axis)) ** (1.0 / order)
        if out is None:
            out = res
        else:
            out._tensor = res._tensor
            out.shape = res.shape
            # TODO: decide how we want to handle differing dtypes
            out.dtype = res.dtype
        return out

    def xcov(self, a, b, out):
        cudanet.xcov(a._tensor, b._tensor, out._tensor)

    def mean_norm(self, a, axis, out):
        a._tensor.mean_norm(axis, out._tensor)

    def exp(self, x, out):
        cudanet.exp(x._tensor, out._tensor)

    def log(self, x, out):
        cudanet.log(x._tensor, out._tensor)

    def logistic(self, x, out):
        cudanet.sigmoid(x._tensor, out._tensor)

    def tanh(self, x, out):
        cudanet.tanh(x._tensor, out._tensor)

    def rectlin(self, x, out):
        self.greater(x, self.wrap(0), out=out)
        self.multiply(x, out, out=out)

    def rectlin_derivative(self, x, out):
        self.greater(x, self.wrap(0), out=out)

    def fill(self, x, val):
        x._tensor[:] = val

    def sum(self, x):
        if x is None:
            return float('NaN')
        return x.sum()

    def mean(self, x):
        if x is None:
            return float('NaN')
        return x.mean()

    def min(self, x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        if axis is None and not keepdims:
            assert out is None
            res = x._tensor.min(axis=0).min(axis=1)
            logger.debug('Copying to host')
            res.copy_to_host()
            return res.numpy_array[0][0]
        if out is None:
            res = x._tensor.min(axis)
        else:
            res = x._tensor.min(axis, out)
        return GPUTensor(res)

    def max(self, x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        if axis is None and not keepdims:
            assert out is None
            res = x._tensor.max(axis=0).max(axis=1)
            logger.debug('Copying to host')
            res.copy_to_host()
            return res.numpy_array[0][0]
        if out is None:
            res = x._tensor.max(axis)
        else:
            res = x._tensor.max(axis, out)
        return GPUTensor(res)

    def argmin(self, tsr, axis, out):
        """
        Calculates the indices of the minimal element value along the specified
        axis.  If multiple elements contain the minimum, only the elements of
        the first are returned.

        Arguments:
            tsr (GPUTensor): The GPUTensor on which to find the minimum indices
            axis (int): The dimension along which to find the minimum.  If set
                        to None, find the overall minimum index of a flattened
                        representation of tsr.
            out (GPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape

        Returns:
            GPUTensor: reference to out
        """
        tsr._tensor.argmin(axis, target=out._tensor)
        return out

    def argmax(self, tsr, axis, out):
        """
        Calculates the indices of the maximal element value along the specified
        axis.  If multiple elements contain the maximum, only the elements of
        the first are returned.

        Arguments:
            tsr (GPUTensor): The GPUTensor on which to find the maximum indices
            axis (int): The dimension along which to find the maximum.  If set
                        to None, find the overall maximum index of a flattened
                        representation of tsr.
            out (GPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape

        Returns:
            GPUTensor: reference to out
        """
        tsr._tensor.argmax(axis, target=out._tensor)
        return out

    def fabs(self, x, out=None):
        if out is not None:
            res = cudanet.abs(x._tensor, out._tensor)
        else:
            # XXX: temporary fix.
            res = cudanet.abs(x._tensor, cudanet.empty(x.shape))
        return GPUTensor(res)

    def sqrt(self, x, out):
        res = cudanet.sqrt(x._tensor, out._tensor)
        return GPUTensor(res)

    def squish(self, obj, n):
        assert obj.shape[0] % n == 0
        return obj.reshape((obj.shape[1] * n, obj.shape[0] / n))

    def softmax(self, x, out):
        cudanet.softmax(x._tensor, out._tensor)

    def softmax_gradient(self, y, err, out):
        cudanet.softmax_grad(y._tensor, err._tensor, out._tensor)

    def nonzero(self, x):
        res = x._tensor.copy()
        res.equals(0)
        res.equals(0)
        return GPUTensor(res)

    def fprop_conv(self, weights, inputs, outputs, links, ifmshape, ofmshape,
                   ofmlocs, padding, stride, nifm, ngroups, prodbuf):
        assert ifmshape[0] == ifmshape[1]
        cudanet.convolution(
            weights._tensor, inputs._tensor, outputs._tensor,
            ifmshape[0], ofmshape[0], ofmshape[1], padding, stride, nifm,
            ngroups)

    def bprop_conv(self, weights, error, berror, links,  ifmshape, ofmshape,
                   ofmlocs, padding, stride, nifm, ngroups, bpropbuf):
        cudanet.deconvolve_errors(
            weights._tensor, error._tensor,
            berror._tensor, ifmshape[0], ifmshape[1], ofmshape[0],
            padding, stride, nifm, ngroups)

    def update_conv(self, weights, inputs, error, updates, links, ifmshape,
                    ofmshape, ofmlocs, padding, stride, nifm, ngroups, fwidth,
                    updatebuf):
        cudanet.deconvolve_wts(
            error._tensor, inputs._tensor, updates._tensor,
            ifmshape[0], ofmshape[0], ofmshape[1], fwidth,
            padding, stride, nifm, ngroups, ofmshape[0])

    def fprop_unpool(self, inputs, outputs, outputsbuf, links,
                     ifmshape, ofmshape, fshape, padding, stride,
                     nfm, maxinds):
        cudanet.unpool_forward(
            smallMat=inputs._tensor, largeMat=outputs._tensor,
            channels=nfm, sizeX=fshape[1], smallX=ifmshape[1],
            largeX=ofmshape[1])

    def bprop_unpool(self, inputs, outputs, error, berror, berrorbuf,
                     links, ifmshape, ofmshape, fshape, padding, stride,
                     nfm, maxinds):
        cudanet.unpool_backward(
            largeMat=inputs._tensor, smallMat=outputs._tensor,
            channels=fshape[1], sizeX=fshape[1], smallX=ifmshape[1],
            largeX=ofmshape[1])

    def fprop_mpool(self, inputs, outputs, outputsbuf, links,
                    ifmshape, ofmshape, fshape, padding, stride, nfm, maxinds):
        cudanet.max_pool(
            inputs._tensor, outputs._tensor, nfm, fshape[1],
            padding, stride, ofmshape[1])

    def bprop_mpool(self, inputs, outputs, error, berror, berrorbuf, links,
                    ifmshape, ofmshape, fshape, padding, stride, nfm, maxinds):
        cudanet.max_pool_undo(
            inputs._tensor, error._tensor, outputs._tensor,
            berror._tensor, fshape[1], padding, stride, ofmshape[1])

    def fprop_cmrnorm(self, inputs, outputs, ifmshape, nfm, ksize, alpha,
                      beta):
        cudanet.crossmap_response_norm(
            inputs._tensor, outputs._tensor, nfm, ksize, alpha, beta)

    def bprop_cmrnorm(self, inputs, outputs, error, berror, ifmshape, nfm,
                      ksize, alpha, beta, tempbuf):
        cudanet.crossmap_response_norm_undo(
            inputs._tensor, error._tensor, outputs._tensor,
            berror._tensor, nfm, ksize, alpha, beta)

    def fprop_apool(self, inputs, outputs, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm):
        cudanet.avg_pool(
            imgs=inputs._tensor, target=outputs._tensor, channels=nfm,
            sizeX=fshape[0], paddingStart=padding, moduleStride=stride,
            numModulesX=ofmshape[0])

    def bprop_apool(self, outputs, error, berror, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm):
        cudanet.avg_pool_undo(
            avgGrads=error._tensor, target=berror._tensor, sizeX=fshape[0],
            paddingStart=padding, moduleStride=stride,
            numModulesX=ofmshape[0], imgSizeX=ifmshape[0])

    def fprop_l2pool(self, inputs, outputs, links, ifmshape, ofmshape,
                     fshape, padding, stride, nfm):
        cudanet.l2_pool(
            imgs=inputs._tensor, target=outputs._tensor, channels=nfm,
            sizeX=fshape[0], paddingStart=padding, moduleStride=stride,
            numModulesX=ofmshape[0])

    def bprop_l2pool(self, inputs, outputs, error, berror, links, ifmshape,
                     ofmshape, fshape, padding, stride, nfm, prodbuf):
        cudanet.l2_pool_undo(
            imgs=inputs._tensor, l2Grads=error._tensor, l2Acts=outputs._tensor,
            target=berror._tensor, sizeX=fshape[0], paddingStart=padding,
            moduleStride=stride, numModulesX=ofmshape[0])

    def fprop_fc(self, inputs, weights, out):
        cudanet.dot(weights._tensor, inputs._tensor, out._tensor)

    def bprop_fc(self, deltas, weights, out):
        cudanet.dot(weights.transpose()._tensor, deltas._tensor, out._tensor)

    def update_fc(self, deltas, inputs, out):
        cudanet.dot(deltas._tensor, inputs.transpose()._tensor, out._tensor)

    def fprop_cmpool(self, inputs, weights, fmsize, out):
        raise NotImplementedError("TODO!")

    def bprop_cmpool(self, deltas, weights, fmsize, out):
        raise NotImplementedError("TODO!")

    def update_cmpool(self, deltas, inputs, fmsize, updatebuf, out):
        raise NotImplementedError("TODO!")

    def sync_stream(self):
        cudanet.sync_stream()

    def gen_weights(self, size, weight_params, dtype=None):
        """
        Different types of weight initializations.  Includes:
        * uniform - uniform distribution
        * sparse_eigenvalued - each weight has 15 nonzero inputs and the
        maximum eigenvalue of the weight matrix is scaled to 1.2
        * normal or gaussian - normal distribution
        * node_normalized - initialization is as discussed in Glorot2010

        Arguments:
            size: shape of the weight matrix to generate
            weight_params: parameters 'type', 'high', 'low', 'loc', etc.

        Returns:
            GPUTensor: The initialized weights
        """
        # FIXME: Get rid of duplication.
        weights = None
        if weight_params['type'] == 'uniform':
            low = 0.0
            high = 1.0
            if 'low' in weight_params:
                low = weight_params['low']
            if 'high' in weight_params:
                high = weight_params['high']
            logger.info('generating %s uniform(%0.2f, %0.2f) weights.' %
                        (str(size), low, high))
            weights = numpy.random.uniform(low, high, size)
        elif (weight_params['type'] == 'gaussian' or
              weight_params['type'] == 'normal'):
            loc = 0.0
            scale = 1.0
            if 'loc' in weight_params:
                loc = weight_params['loc']
            if 'scale' in weight_params:
                scale = weight_params['scale']
            logger.info('generating %s normal(%0.2f, %0.2f) weights.' %
                        (str(size), loc, scale))
            weights = numpy.random.normal(loc, scale, size)
        elif (weight_params['type'] == 'sparse_eigenvalued'):
            # initialization for RNNS as in Sutskever 2013
            sparseness = 15
            eigenvalue = 1.2
            if 'sparseness' in weight_params:
                sparseness = weight_params['sparseness']
            if 'eigenvalue' in weight_params:
                eigenvalue = weight_params['eigenvalue']
            logger.info('generating %s SI-EV(%0.2f, %0.2f) weights.' %
                        (str(size), sparseness, eigenvalue))
            elements = size[0]*size[1]
            nonzeros = size[0] * sparseness
            weights = numpy.zeros(size).flatten()
            nonzeroindex = numpy.random.permutation(elements)[0:nonzeros]
            weights[nonzeroindex] = 0.3 * numpy.random.randn(nonzeros)
            weights = weights.reshape(size).copy()
            if size[0] == size[1]:
                temp = numpy.linalg.eig(weights)
                max_eig = numpy.max(numpy.absolute(temp[0]))
                logger.info('cpu: dividing by max eigenvalue %2.2f', max_eig)
                weights = eigenvalue * weights / max_eig
            else:
                logger.info('Matrix is non-square, no eigenvalue scaling.')
        elif weight_params['type'] == 'node_normalized':
            # initialization is as discussed in Glorot2010
            scale = 1.0
            if 'scale' in weight_params:
                scale = weight_params['scale']
            logger.info('generating %s node_normalized(%0.2f) weights.' %
                        (str(size), scale))
            node_norm = scale * math.sqrt(6.0 / sum(size))
            weights = numpy.random.uniform(-node_norm, node_norm, size)
        else:
            raise AttributeError("invalid weight_params specified")
        if 'bias_init' in weight_params:
            # per append_bias() bias weights are in the last column
            logger.info('separately initializing bias weights to %0.2f' %
                        weight_params['bias_init'])
            weights[:, -1] = weight_params['bias_init']

        return GPUTensor(numpy.array(weights, numpy.float32))


class GPUDataDist(GPU):

    """
    helper sub-class for data parallel implementations
    """

    def __init__(self, **kwargs):
        local_rank = numpy.int32(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        local_size = numpy.int32(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        num_devices = cudanet.get_num_devices()

        if (local_size > num_devices):
            logger.warning('Node %s: requested device: %d  max devices: %d' %
                           (MPI.Get_processor_name(), local_size,
                            num_devices))
            raise AttributeError("Asking for more gpu devices than are "
                                 "available on node")

        cudanet.set_device_id(local_rank)
        logger.info('Setting Device on %s to %d' %
                    (MPI.Get_processor_name(), local_rank))
        super(GPUDataDist, self).__init__(**kwargs)

    def update_fc_dot(self, deltas, inputs, out):
        super(GPUDataDist, self).update_fc_dot(deltas, inputs, out)
        # trivial implementation below
        # could optimize by making each proc responsible for #params/comm.size
        # of the params
        out.raw(False)[:] = MPI.COMM_WORLD.reduce(out.raw(), op=MPI.SUM,
                                                  root=0)
        out.raw(False)[:] = MPI.COMM_WORLD.bcast(out.raw(False))

        out.copy_to_device()

    def update_conv(self, weights, inputs, error, updates, links, ifmshape,
                    ofmshape, ofmlocs, padding, stride, nifm, ngroups, fwidth,
                    updatebuf):
        super(GPUDataDist, self).update_conv(weights, inputs, error, updates,
                                             links, ifmshape, ofmshape,
                                             ofmlocs, padding, stride, nifm,
                                             ngroups, fwidth, updatebuf)

        updates.raw(False)[:] = MPI.COMM_WORLD.reduce(updates.raw(),
                                                      op=MPI.SUM, root=0)
        updates.raw(False)[:] = MPI.COMM_WORLD.bcast(updates.raw(False))

        updates.copy_to_device()
