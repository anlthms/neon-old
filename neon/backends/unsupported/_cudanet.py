"""
A `cuda-convnet2 <https://code.google.com/p/cuda-convnet2/>`_ GPU based backend.
"""

import logging
import numpy
import math
import cudanet

from neon.backends.backend import Backend, Tensor
from neon.util.error import TooSlowToImplementError

logger = logging.getLogger(__name__)


class CudanetTensor(Tensor):

    """
    Simple wrapped `cudanet.CUDAMatrix` tensor

    Arguments:
        obj (numpy.ndarray): the actual data values (will be converted
                             to 2-d row matrix).  Python built-in types like
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
        return self._tensor.asarray()

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
            CudanetTensor: view or new sliced copy

        Raises:
            TooSlowToImplementError: if invalid `_slice` provided (too
            complex to implement quickly).
        """
        res = self
        fn = res._tensor.get_row_slice
        if dim == 1:
            fn = res._tensor.get_col_slice
        if isinstance(_slice, int):
            _slice = slice(_slice, _slice + 1)
        if isinstance(_slice, slice):
            assert _slice.step is None or _slice.step == 1
            start, stop, stride = _slice.indices(self.shape[dim])
            res = CudanetTensor(fn(start, stop))
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
            CudanetTensor: view or new sliced copy

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
            value (CudanetTensor, int, float): The values to assign at each
                                               key position.  Must be scalar
                                               or if a CudanetTensor, must
                                               have the right shape.

        Returns:
            CudanetTensor: update view of this tensor

        Raises:
            IndexError: if invalid number of dimensions specified in key.
            NotImplementedError: if invalid value type passed.
            TooSlowToImplementError: if arbitrarily indexed key passed.
        """
        if isinstance(value, CudanetTensor):
            value = value._tensor
        elif not isinstance(value, (int, float)):
            raise NotImplementedError("can only assign Cudanet tensors or "
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
            # 1-D index, check for form x[:] = value
            if isinstance(key, slice):
                start, stop, stride = key.indices(self.shape[0])
                if start == 0 and stop == self.shape[0]:
                    self._tensor.assign(value)
                else:
                    raise IndexError("1-D partial indexing unsupported")
            else:
                raise IndexError("Invalid 1-D index type")

    def __delitem__(self, key):
        raise ValueError("cannot delete array elements")

    def __float__(self):
        raise NotImplementedError()

    def __neg__(self):
        return -1 * self

    def __lt__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, CudanetTensor):
            self._tensor.less_than(other._tensor, target)
        else:
            self._tensor.less_than(other, target)
        return CudanetTensor(target)

    def __le__(self, other):
        # call __lt__ and __eq__ and iterate?
        raise NotImplementedError()

    def __eq__(self, other):
        if other is None:
            return False
        target = cudanet.empty(self.shape)
        if isinstance(other, CudanetTensor):
            self._tensor.equals(other._tensor, target)
        else:
            self._tensor.equals(other, target)
        return CudanetTensor(target)

    def __ne__(self, other):
        # go through results of __eq__ and negate
        raise NotImplementedError()

    def __gt__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, CudanetTensor):
            self._tensor.greater_than(other._tensor, target)
        else:
            self._tensor.greater_than(other, target)
        return CudanetTensor(target)

    def __ge__(self, other):
        # call __gt__ and __eq__ and iterate?
        raise NotImplementedError()

    def __add__(self, other):
        """
        Perform element-wise addition with the items in other.
        Now supports limited broadcasting to add a vector to a matrix.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same
                            dimensions as this Tensor, or be broadcastable
                            as such.

        Returns:
            CudanetTensor: containing the element-wise sum values.
        """

        if self.shape == other.shape:
            target = cudanet.empty(self.shape)
            if isinstance(other, CudanetTensor):
                self._tensor.add(other._tensor, target)
            else:
                self._tensor.add(other, target)
            return CudanetTensor(target)
        else:
            if other.shape[1] == 1:  # [Nx1] vector
                ones = cudanet.empty((self.shape[0], 1))
                ones.assign(1)
                # outer product repmat (probably quite inefficient)
                other = CudanetTensor(cudanet.dot(ones, other._tensor.T))
            else:  # [1xN] vector
                ones = cudanet.empty((self.shape[0], 1))
                ones.assign(1)
                other = CudanetTensor(cudanet.dot(ones, other._tensor))
            target = cudanet.empty(self.shape)
            if isinstance(other, CudanetTensor):
                self._tensor.add(other._tensor, target)
            else:
                self._tensor.add(other, target)
            return CudanetTensor(target)

    def __radd__(self, other):
        """
        Perform element-wise addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same
                            dimensions as this Tensor, or be broadcastable
                            as such.

        Returns:
            CudanetTensor: containing the element-wise sum values.
        """
        target = cudanet.empty(self.shape)
        if isinstance(other, CudanetTensor):
            self._tensor.add(other._tensor, target)
        else:
            self._tensor.add(other, target)
        return CudanetTensor(target)

    def __iadd__(self, other):
        """
        Perform element-wise in-place addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same
                            dimensions as this Tensor, or be broadcastable
                            as such.

        Returns:
            CudanetTensor: updated view of this Tensor
        """
        if isinstance(other, CudanetTensor):
            self._tensor.add(other._tensor)
        else:
            self._tensor.add(other)
        return self

    def __sub__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, CudanetTensor):
            self._tensor.subtract(other._tensor, target)
        else:
            self._tensor.subtract(other, target)
        return CudanetTensor(target)

    def __rsub__(self, other):
        target = cudanet.empty(self.shape)
        self._tensor.mult(-1.0, target)
        if isinstance(other, CudanetTensor):
            target.add(other._tensor)
        else:
            target.add(other)
        return CudanetTensor(target)

    def __isub__(self, other):
        if isinstance(other, CudanetTensor):
            self._tensor.subtract(other._tensor)
        else:
            self._tensor.subtract(other)
        return self

    def __mul__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, CudanetTensor):
            self._tensor.mult(other._tensor, target)
        else:
            self._tensor.mult(other, target)
        return CudanetTensor(target)

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        if isinstance(other, CudanetTensor):
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
        if isinstance(other, CudanetTensor):
            self._tensor.divide(other._tensor, target)
        else:
            self._tensor.divide(other, target)
        return CudanetTensor(target)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rtruediv__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, (float, int)):
            other = CudanetTensor(other * numpy.ones(self.shape))
        if isinstance(other, CudanetTensor):
            other._tensor.divide(self._tensor, target)
        elif isinstance(other, cudanet.CUDAMatrix):
            other.divide(self._tensor, target)
        else:
            return NotImplemented
        return CudanetTensor(target)

    def __idiv__(self, other):
        if isinstance(other, CudanetTensor):
            self._tensor.divide(other._tensor)
        else:
            self._tensor.divide(other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, CudanetTensor):
            self._tensor.divide(other._tensor)
        else:
            self._tensor.divide(other)
        return self

    def __pow__(self, other, modulo=None):
        target = cudanet.empty(self.shape)
        if isinstance(other, CudanetTensor):
            cudanet.pow(self._tensor, other._tensor, target)
        else:
            cudanet.pow(self._tensor, other, target)
        return CudanetTensor(target)

    def __rpow__(self, other):
        target = cudanet.empty(self.shape)
        if isinstance(other, (float, int)):
            other = CudanetTensor(other)
        if isinstance(other, CudanetTensor):
            cudanet.pow(other._tensor, self._tensor, target)
        elif isinstance(other, cudanet.CUDAMatrix):
            cudanet.pow(other, self._tensor, target)
        else:
            return NotImplemented
        return CudanetTensor(target)

    def __ipow__(self, other):
        if isinstance(other, CudanetTensor):
            cudanet.pow(self._tensor, other._tensor)
        else:
            cudanet.pow(self._tensor, other)
        return self

    def copy(self):
        return CudanetTensor(self._tensor.copy())

    def raw(self):
        self._tensor.copy_to_host()
        return self._tensor.numpy_array

    def T(self): # flake8: noqa
        # CUDAMatrix.T is a transposed view.
        return TransposedCudanetTensor(self._tensor, self._tensor.T)

    def transpose(self):
        # CUDAMatrix.transpose() returns a transposed copy.
        return CudanetTensor(self._tensor.transpose())

    def reshape(self, shape):
        return CudanetTensor(self._tensor.reshape(shape))

    def argmax(self, axis):
        return CudanetTensor(self._tensor.argmax(axis))

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
                return CudanetTensor(self._tensor.get_row_slice(indices[0],
                                                                indices[-1] +
                                                                1))
            elif axis == 1:
                return CudanetTensor(self._tensor.get_col_slice(indices[0],
                                                                indices[-1] +
                                                                1))
            elif axis is None:
                # we might be able to do this by first doing a reshape?
                raise TooSlowToImplementError("need to first reshape")
        else:
            raise TooSlowToImplementError("CUDAMatrix can't do arbitrary"
                                          " indexing efficiently")

    def add(self, obj):
        self._tensor.add(obj._tensor)

    def sub(self, obj):
        self._tensor.subtract(obj._tensor)

    def sum(self, axis=None):
        """
        Sum elements of a CudanetTensor. If axis is None, all elements are
        summed and a numpy scalar returned. If axis is 1 or 2, sum along that
        axis and return a CudanetTensor.
        """
        if axis is None:
            result = self._tensor.sum(axis=0).sum(axis=1)
            logger.debug('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]
        else:
            result = self._tensor.sum(axis=axis)
            logger.debug('major change in functionality of sum')
            return CudanetTensor(result)

    def mean(self):
        result = self._tensor.mean(axis=0).mean(axis=1)
        logger.debug('Copying to host')
        result.copy_to_host()
        return result.numpy_array[0][0]

    def min(self):
        result = self._tensor.min(axis=0).min(axis=1)
        logger.debug('Copying to host')
        result.copy_to_host()
        return result.numpy_array[0][0]

    def max(self):
        result = self._tensor.max(axis=0).max(axis=1)
        logger.debug('Copying to host')
        result.copy_to_host()
        return result.numpy_array[0][0]

    def log(self):
        target = cudanet.empty(self.shape)
        cudanet.log(self._tensor, target)
        return CudanetTensor(target)

    def exp(self):
        target = cudanet.empty(self.shape)
        cudanet.exp(self._tensor, target)
        return CudanetTensor(target)

    def get_minor_slice(self, start, end):
        return self.__class__(self[:, start:end]._tensor)

    def set_minor_slice(self, start, end, data):
        self[:, start:end] = data

    def get_major_slice(self, start, end):
        return self.__class__(self[start:end]._tensor)

    def set_major_slice(self, start, end, data):
        self[start:end] = data

    def major_axis(self):
        return 1

    def minor_axis(self):
        return 0


class TransposedCudanetTensor(CudanetTensor):
    """
    Transposed CUDAMatrix tensor
    """

    def __init__(self, obj, transposed):
        assert type(obj) == cudanet.CUDAMatrix
        self._tensor = transposed
        self.shape = (obj.shape[1], obj.shape[0])


class Cudanet(Backend):
    """
    A `cuda-convnet2 <https://code.google.com/p/cuda-convnet2/>`_  based
    backend for matrix operations.

    Attributes:
        epsilon (float): the unit roundoff for the elements underlying this
                         tensor.
    """
    # we need to cast epsilon to float to ensure it works with some of the type
    # checking in cudanet functions like less_than() and so forth
    epsilon = float(numpy.finfo(numpy.float32).eps)
    default_dtype = numpy.float32
    tensor_cls = CudanetTensor

    def __init__(self, **kwargs):
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
        return CudanetTensor(cudanet.empty(shape))

    def zeros(self, shape, dtype=numpy.float32):
        return CudanetTensor(cudanet.CUDAMatrix(
            numpy.zeros(shape, dtype=dtype)))

    def alloc(self, nrows, ncols, dtype=numpy.float32):
        return CudanetTensor(cudanet.CUDAMatrix(
            numpy.zeros((ncols, nrows), dtype=dtype)))

    def ones(self, shape, dtype=numpy.float32):
        return CudanetTensor(cudanet.CUDAMatrix(
            numpy.ones(shape, dtype=dtype)))

    def array(self, obj, dtype=None):
        ndarray = numpy.array(obj, dtype=numpy.float32)
        if ndarray.ndim == 1:
            ndarray = ndarray.reshape((1, ndarray.shape[0]))
        return CudanetTensor(ndarray)

    def wrap(self, obj):
        return CudanetTensor(obj)

    def clip(self, a, a_min, a_max, out=None):
        if out is None:
            out = CudanetTensor(cudanet.empty((a.shape[0], a.shape[1])))
        # storage needed here is pretty atrocious.  Any way we could speed this
        # up?  Would iterating element wise be faster?
        clip_mask = cudanet.empty((a.shape[0], a.shape[1]))
        clip_vals = cudanet.empty((a.shape[0], a.shape[1]))
        # clip values < a_min to a_min in out
        a._tensor.less_than(a_min, clip_mask)
        clip_vals.assign(a_min)
        cudanet.where(clip_mask, clip_vals, a._tensor, out._tensor)
        # clip values > a_max to a_max in out
        out._tensor.greater_than(a_max, clip_mask)
        clip_vals.assign(a_max)
        cudanet.where(clip_mask, clip_vals, out._tensor, out._tensor)
        return out

    def rng_init(self):
        seed = None
        if 'rng_seed' in self.__dict__:
            seed = self.rng_seed
        numpy.random.seed(seed)
        try:
            cudanet.CUDAMatrix.init_random(seed)
        except TypeError:
            if seed is not None:
                logger.warn("Must seed random number generator with an "
                            "integer.  You specified: %s" % str(seed))
            cudanet.CUDAMatrix.init_random(0)

    def uniform(self, low=0.0, high=1.0, size=1):
        seq = numpy.random.uniform(low, high, size)
        return CudanetTensor(numpy.array(seq, dtype=numpy.float32))

    def normal(self, loc=0.0, scale=1.0, size=1):
        seq = numpy.random.normal(loc, scale, size)
        return CudanetTensor(numpy.array(seq, dtype=numpy.float32))

    def append_bias(self, x):
        """
        Adds a bias row to CudanetTensor x, returning a new CudanetTensor.
        """
        result = cudanet.empty((x.shape[0] + 1, x.shape[1]))
        result.set_row_slice(0, x.shape[0], x._tensor)
        result.set_row_slice(x.shape[0], (x.shape[0] + 1),
                             cudanet.CUDAMatrix.ones.slice(
                                0, x.shape[1]).reshape((1, x.shape[1])))
        return CudanetTensor(result)

    def copy(self, a):
        assert type(a) == CudanetTensor
        return a.copy()

    def argmax(self, x, axis=None):
        return CudanetTensor(x._tensor.argmax(axis))

    def dot(self, a, b, out):
        cudanet.dot(a._tensor, b._tensor, out._tensor)

    def add(self, a, b, out):
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

    def greater(self, a, b, out):
        a._tensor.greater_than(b._tensor, out._tensor)

    def exp(self, x, out):
        cudanet.exp(x._tensor, out._tensor)

    def log(self, x, out):
        cudanet.log(x._tensor, out._tensor)

    def logistic(self, x, out):
        cudanet.sigmoid(x._tensor, out._tensor)

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
            res = cudanet.min(x._tensor, axis)
        else:
            res = cudanet.min(x._tensor, axis, out)

        return CudanetTensor(res)

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
            res = cudanet.max(x._tensor, axis)
        else:
            res = cudanet.max(x._tensor, axis, out)

        return CudanetTensor(res)

    def fabs(self, x, out=None):
        if out is not None:
            res = cudanet.abs(x._tensor, out._tensor)
        else:
            # XXX: temporary fix.
            res = cudanet.abs(x._tensor, cudanet.empty(x.shape))
        return CudanetTensor(res)

    def sqrt(self, x, out):
        res = cudanet.sqrt(x._tensor, out._tensor)
        return CudanetTensor(res)

    def square(self, x, out):
        res = cudanet.apply_pow(x._tensor, 2.0, out._tensor)
        return CudanetTensor(res)

    def cube(self, x, out):
        res = cudanet.apply_pow(x._tensor, 3.0, out._tensor)
        return CudanetTensor(res)

    def squish(self, obj, n):
        assert obj.shape[0] % n == 0
        return obj.reshape((obj.shape[1] * n, obj.shape[0] / n))

    def not_equal(self, x, y):
        res = x._tensor.copy()
        res.equals(y._tensor)
        res.equals(0)
        return CudanetTensor(res)

    def nonzero(self, x):
        res = x._tensor.copy()
        res.equals(0)
        res.equals(0)
        return CudanetTensor(res)

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

    def fprop_mpool(self, inputs, outputs, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm, maxinds):
        cudanet.max_pool(
            inputs._tensor, outputs._tensor, nfm, fshape[1],
            padding, stride, ofmshape[1])

    def bprop_mpool(self, inputs, outputs, error, berror, links, ifmshape,
                    ofmshape, fshape, padding, stride, nfm, maxinds):
        cudanet.max_pool_undo(
            inputs._tensor, error._tensor, outputs._tensor,
            berror._tensor, fshape[1], padding, stride, ofmshape[1])

    def fprop_apool(self, inputs, outputs, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm):
        raise NotImplementedError("TODO!")

    def bprop_apool(self, outputs, error, berror, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm):
        raise NotImplementedError("TODO!")

    def fprop_l2pool(self, inputs, outputs, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm):
        raise NotImplementedError("TODO!")

    def bprop_l2pool(self, outputs, error, berror, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm, prodbuf):
        raise NotImplementedError("TODO!")

    def fprop_fc_dot(self, inputs, weights, out):
        cudanet.dot(weights._tensor, inputs._tensor, out._tensor)

    def bprop_fc_dot(self, deltas, weights, out):
        cudanet.dot(weights.T()._tensor, deltas._tensor, out._tensor)

    def update_fc_dot(self, deltas, inputs, out):
        cudanet.dot(deltas._tensor, inputs.T()._tensor, out._tensor)

    def format(self, raw):
        return self.array(raw.T.copy())

    def gen_weights(self, size, weight_params, dtype=None):
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

        return CudanetTensor(numpy.array(weights, numpy.float32))

    def get_momentum_coef(self, epoch, momentum_params):
        # FIXME: Get rid of duplication.
        coef = 0.0
        if 'coef' in momentum_params:
            coef = momentum_params['coef']
        if 'initial_coef' in momentum_params:
            init_coef = momentum_params['initial_coef']
        else:
            init_coef = coef
        if 'saturated_coef' in momentum_params:
            saturated_coef = momentum_params['saturated_coef']
        else:
            saturated_coef = coef
        if 'start_epoch' in momentum_params:
            start_epoch = momentum_params['start_epoch']
        else:
            start_epoch = None
        if 'saturate_epoch' in momentum_params:
            saturate_epoch = momentum_params['saturate_epoch']
        else:
            saturate_epoch = None

        if momentum_params['type'] == 'constant':
            pass
        elif momentum_params['type'] == 'linear_monotone':
            coef = init_coef
            if start_epoch is not None and epoch >= start_epoch:
                if saturate_epoch is not None and epoch <= saturate_epoch:
                    if start_epoch == saturate_epoch:
                        coef = saturated_coef
                    else:
                        init_proportion = ((epoch - start_epoch + 0.0) /
                                           (saturate_epoch - start_epoch))
                        coef = (init_proportion * init_coef +
                                (1.0 - init_proportion) * saturated_coef)
                elif saturate_epoch is not None and epoch > saturate_epoch:
                    coef = saturated_coef
            else:
                coef = saturated_coef
        elif momentum_params['type'] == 'nesterov':
            raise NotImplementedError("TODO!")
        else:
            raise AttributeError("invalid momentum_params specified")
        return coef
