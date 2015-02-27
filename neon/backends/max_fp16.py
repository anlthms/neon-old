# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
16bit half-precision floating point backend for Maxwell kernels.
So far this is a stripped down cudanet to check which functions are needed
"""

# import cudanet
from nervana_lib import NervanaLib as nl
import logging
import numpy as np

from neon.backends.backend import Backend, Tensor
from neon.util.compat import range
from neon.util.error import TooSlowToImplementError

logger = logging.getLogger(__name__)


class GPUTensor(Tensor):

    """
    Our n-dimensional array data structure that can reside on host or on GPU
    device.  Our implementation is a wrapped `cudanet.CUDAMatrix` tensor, where
    cudanet is derived from cuda-convnet2.

    Arguments:
        obj (numpy.ndarray): The actual data values (will be converted
                             to a 2-d row matrix).  Python built-in types like
                             lists and tuples are also supported.
        dtype (None, optional): Underlying data type of the elements.
                                Ignored for this backend as all values are
                                stored in cudanet as float32's.

    Notes:
        This implementation currently has the following limitations:
        * only 2D shaped Tensors are supported (set in _min_dims)
        * All element values are stored as float32 (input may be converted if
          input of a differing type is passed)
        * Only contiguous rectangular slicing is supported.  Sliced assignment
          can only be done along a singular subsetted dimension (i.e. only row
          slice *or* column slice based assignment).
    """
    _tensor = None
    _min_dims = 2

    def __init__(self, obj, dtype=None, copy_to_device=True):
        if type(obj) == cudanet.CUDAMatrix:
            self._tensor = obj
            self.shape = self._tensor.shape
        else:
            if type(obj) == list:
                obj = numpy.array(obj)
            if isinstance(obj, numpy.ndarray):
                # CUDAMatrix only supports ndarrays with exactly 2 dimensions
                # (though the elements can be tuples/lists to create arbitrary
                # n dimensions)
                while obj.ndim < self._min_dims:
                    obj = obj.reshape(obj.shape + (1, ))
                if obj.ndim != self._min_dims:
                    raise ValueError("CUDAMatrix only supports %d-D"
                                     "matrices.  You specifed %d-D" %
                                     (self._min_dims, obj.ndim))
                logger.debug('Copying to GPU')
                if dtype not in (numpy.float32, numpy.int32, 'float32',
                                 'int32') or dtype is None:
                    logger.debug('dtype %s is unsupported in GPU '
                                 'backend, defaulting to float32', dtype)
                    obj = numpy.array(obj, dtype='float32')
                elif obj.dtype != dtype:
                    logger.debug('object dtype %s mismatch.  '
                                 'Converting to %s', obj.dtype, dtype)
                    obj = numpy.array(obj, dtype=dtype)
                self._tensor = cudanet.CUDAMatrix(obj)
                self.shape = self._tensor.shape
            else:
                self._tensor = obj
        self.dtype = dtype

    @property
    def raw(self):
        return self._tensor

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
            raise TooSlowToImplementError("slice indexing too complex")
        return res

    def asnumpyarray(self):
        """
        Convert the GPUTensor to an in host memory `numpy.ndarray`.  A copy of
        the data may be made depending on where the GPUTensor normally resides.

        Returns:
            numpy.ndarray view or copy of the GPUTensor data.
        """
        self._tensor.copy_to_host()
        return self._tensor.numpy_array

    def __getitem__(self, key):
        """
        Extract a subset view of the items via slice style indexing
        along each dimension. e.g. A[5:10, :].  Each slice consists of
        start_idx:stop_idx:step_size triplets.  If step_size isn't specified it
        defaults to 1.  If start_idx isn't specified it defaults to 0.  If
        stop_idx isn't specified it defaults to the total number of elements
        along that dimension.  As such a slice value of ':' allows one to
        select all elements along that dimension.

        Arguments:
            key (int, slice, tuple): indices of each dimension's slice.

        Returns:
            GPUTensor: view of self corresponding to the subset items.

        Raises:
            IndexError: if invalid number of dimensions specified in key.

        See Also:
            take
        """
        res = self
        if isinstance(key, tuple):
            if len(key) > self._min_dims:
                raise IndexError("CUDAMatrix only supports %d-D matrices",
                                 self._min_dims)
            else:
                for idx in range(len(key) - 1, -1, -1):
                    res = res._slice_dim(key[idx], idx)
        else:
            res = res._slice_dim(key, 0)
        return res

    def __setitem__(self, key, value):
        """
        Assign the specified value to a subset of elements found via slice
        style indexing along each dimension. e.g. A[5:10, :] = 4.5.
        Each slice consists of start_idx:stop_idx:step_size triplets.  If
        step_size isn't specified it defaults to 1.  If start_idx isn't
        specified it defaults to 0.  If stop_idx isn't specified it defaults
        to the total number of elements along that dimension.  As such a slice
        value of ':' allows one to select all elements along that dimension.

        Arguments:
            key (int, slice, tuple): indices of each dimension's slice.
            value (numeric array, GPUTensor): values to be assigned to the
                                              extracted element subset.  If an
                                              array it should be the same shape
                                              as what key indexes (or be
                                              broadcastable as such).

        Raises:
            IndexError: if invalid number of dimensions specified in key.
            ValueError: if invalid value type passed.
            TooSlowToImplementError: if arbitrarily indexed key passed.

        Notes:
            Currently, this implementation only supports assignment in which
            only a single dimension is subset.  That is, for a 4x4 matrix A,
            assignment to A[1:3, :] and A[:, 1:3] are ok, but A[1:3, 1:3] is
            not.  Attempts to perform such assignment will raise a
            TooSlowToImplementError.
        """
        if isinstance(value, GPUTensor):
            value = value._tensor
        elif not isinstance(value, (int, float)):
            raise ValueError("can only assign GPUTensor's or numeric scalars")
        if isinstance(key, tuple):
            if len(key) > self._min_dims:
                raise IndexError("CUDAMatrix only supports %d-D matrices",
                                 self._min_dims)
            elif len(key) == self._min_dims:
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

    def set_host_mat(self, newarray):
        """
        Changes the host pointer for this tensor to point to a new numpy array
        and its associated data. newarray must be a numpy array
        """
        self._tensor.set_host_mat(newarray)

    def copy_to_device(self):
        self._tensor.copy_to_device()

    def copy_from(self, src):
        """
        Copy contents from src.

        Arguments:
            src (numpy.ndarray): the host-resident object to copy from
        """
        self._tensor.copy_from(src)

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

    def fill(self, value):
        """
        Assign specified value to each element of this CPUTensor.

        Arguments:
            value (numeric): The value to be assigned to each element.

        Return:
            CPUTensor: updated view of the data.
        """
        self._tensor.assign(value)
        return self

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


class MAX_FP16(Backend):

    """
    Stripped down `cuda-convnet2` based backend.
    """
    default_dtype = 'float32'
    tensor_cls = GPUTensor

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.par = None
        if hasattr(self, 'device_id') is False or self.device_id is None:
            self.device_id = 0
        num_devices = cudanet.get_num_devices()
        if self.device_id >= num_devices:
            raise ValueError('Requested device (%d) is unavailable.' %
                             self.device_id)
        cudanet.set_device_id(self.device_id)
        cudanet.cublas_init()
        self.rng_init()

    def default_dtype_if_missing(self, in_dtype):
        if in_dtype is None:
            in_dtype = self.default_dtype
        return in_dtype

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
            shape (int, list): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value ('float32'
                                     unless overridden).

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(cudanet.empty(shape), dtype)

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
                                     use default_dtype value ('float32'
                                     unless overridden).

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        ndarray = numpy.array(obj, dtype=dtype)
        if ndarray.ndim == 1:
            ndarray = ndarray.reshape((1, ndarray.shape[0]))
        return self.tensor_cls(ndarray, dtype)

    def zeros(self, shape, dtype=None):
        """
        Instantiate a new instance of the GPUTensor class setting each element
        value to 0.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value ('float32'
                                     unless overridden).

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(cudanet.CUDAMatrix(numpy.zeros(shape,
                                                              dtype=dtype)),
                               dtype)

    def ones(self, shape, dtype=None):
        """
        Instantiate a new instance of the GPUTensor class setting each element
        value to 1.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value ('float32'
                                     unless overridden).

        Returns:
            GPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(cudanet.CUDAMatrix(numpy.ones(shape,
                                                             dtype=dtype)),
                               dtype)

    def _unwrap(self, obj):
        """
        Helper that extracts and returns the raw data underlying obj (if it is
        a GPUTensor), otherwise returns the existing structure.

        Arguments:
            obj (numeric, GPUTensor): The object to extract raw data from

        Returns:
            numeric, cudanet.CUDAMatrix: raw data from object.
        """
        if isinstance(obj, self.tensor_cls):
            return obj._tensor
        else:
            return obj

    # def copy(self, tsr):

    def clip(self, a, a_min, a_max, out=None):
        if out is None:
            out = self.tensor_cls(cudanet.empty((a.shape[0], a.shape[1])),
                                  self.default_dtype_if_missing(None))
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
                            "integer.  You specified: %s", str(seed))
            cudanet.cudanet_init_random(0)

    def uniform(self, low=0.0, high=1.0, size=1, dtype=None):
        """
        Uniform random number sample generation.

        Arguments:
            low (numeric, optional): Minimal sample value that can be returned.
                                     Defaults to 0.0
            high (numeric, optional): Maximal sample value.  Open ended range
                                      so maximal value slightly less.
                                      Defaults to 1.0
            size (array_like or int, optional): Shape of generated samples
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value ('float32'
                                     unless overridden).

        Returns:
            GPUTensor: Of specified size filled with these random numbers.
        """
        seq = numpy.random.uniform(low, high, size)
        dtype = self.default_dtype_if_missing(None)
        return self.tensor_cls(numpy.array(seq, dtype), dtype)

    # def fill_uniform_thresh(self, tsr, keepthresh=0.5, dtype=None):

    # def normal(self, loc=0.0, scale=1.0, size=1, dtype=None):

    def add(self, left, right, out):
        """
        Perform element-wise addition on the operands left and right, storing
        the result in the GPUTensor out.  Each operand and out is assumed to
        have identical shape, or be broadcastable as such.

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(left, self.tensor_cls):
            left._tensor.add(self._unwrap(right), out._tensor)
        elif isinstance(right, self.tensor_cls):
            right._tensor.add(left, out._tensor)
        else:
            left = self.tensor_cls(left)
            left._tensor.add(right, out._tensor)
        return out

    def subtract(self, left, right, out):
        """
        Perform element-wise subtraction on the operands left and right,
        storing the result in the GPUTensor out.  Each operand and out is
        assumed to have identical shape, or be broadcastable as such.

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(left, self.tensor_cls):
            left._tensor.subtract(self._unwrap(right), out._tensor)
        elif isinstance(right, self.tensor_cls):
            right._tensor.subtract(left, out._tensor)
            out._tensor.mult(-1.0, out._tensor)
        else:
            left = self.tensor_cls(left)
            left._tensor.subtract(right, out._tensor)
        return out

    def multiply(self, left, right, out):
        """
        Perform element-wise multiplication on the operands left and right,
        storing the result in the GPUTensor out.  Each operand and out is
        assumed to have identical shape, or be broadcastable as such.

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(left, self.tensor_cls):
            left._tensor.mult(self._unwrap(right), out._tensor)
        elif isinstance(right, self.tensor_cls):
            right._tensor.mult(left, out._tensor)
        else:
            left = self.tensor_cls(left)
            left._tensor.mult(right, out._tensor)
        return out

    def divide(self, left, right, out):
        """
        Perform element-wise division on the operands left and right, storing
        the result in out.  Each operand and out is assumed to have identical
        shape, or be broadcastable as such.

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if not isinstance(left, self.tensor_cls):
            left = self.tensor_cls(left)
        left._tensor.divide(self._unwrap(right), out._tensor)
        return out

    # def power(self, tsr, power, out):

    # def reciprocal(self, a, out):

    # def dot(self, left, right, out, alpha=1, beta=0):


    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if isinstance(left, self.tensor_cls):
            left._tensor.equals(self._unwrap(right), out._tensor)
        elif isinstance(right, self.tensor_cls):
            right._tensor.equals(left, out._tensor)
        else:
            left = self.tensor_cls(left)
            left._tensor.equals(right, out._tensor)
        return out

    def not_equal(self, left, right, out):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
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
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        if not isinstance(left, self.tensor_cls):
            left = self.tensor_cls(left)
        left._tensor.greater_than(self._unwrap(right), out._tensor)
        return out

    # def greater_equal(self, left, right, out):

    # def less(self, left, right, out):

    # def less_equal(self, left, right, out):

    # def norm(self, tsr, order=None, axis=None, out=None):

    # def xcov(self, a, b, out):

    # def mean_norm(self, a, axis, out):

    def exp(self, x, out):
        cudanet.exp(x._tensor, out._tensor)

    def log(self, x, out):
        cudanet.log(x._tensor, out._tensor)

    def logistic(self, x, out):
        cudanet.sigmoid(x._tensor, out._tensor)

    # def tanh(self, x, out):

    def rectlin(self, x, out):
        # x and out are the same buffer
        cudanet.maximum_scalar(x._tensor, 0., out._tensor)

    def rectlin_derivative(self, x, out):
        self.greater(x, 0, out=out)

    def sum(self, tsr, axes, out):
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
        """
        if isinstance(axes, (tuple, list)):
            logger.warn("GPUTensor only supports single axis for sum.  "
                        "You specified: %s", str(axes))
        else:
            tsr._tensor.sum(axis=axes, target=out._tensor)
        return out

    # def mean(self, tsr, axes, out):

    # def min(self, tsr, axes, out):

    # def max(self, tsr, axes, out):

    # def argmin(self, tsr, axis, out):

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

    # def fabs(self, x, out=None):

    def sqrt(self, x, out):
        res = cudanet.sqrt(x._tensor, out._tensor)
        return GPUTensor(res)

    # def softmax(self, x, out):

    # def softmax_gradient(self, y, err, out):

    # def nonzero(self, x):

    def fprop_fc(self, out, inputs, weights, layer=None):
        """
        Forward propagate the inputs of a fully connected network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            layer (Layer): The layer object.
        """
        cudanet.dot(weights._tensor, inputs._tensor, out._tensor)

    def bprop_fc(self, out, weights, deltas, layer=None):
        """
        Backward propagate the error through a fully connected network layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        cudanet.dot(weights.transpose()._tensor, deltas._tensor, out._tensor)

    def update_fc(self, out, inputs, deltas, layer=None):
        """
        Compute the updated gradient for a fully connected network layer.

        Arguments:
            out (GPUTensor): Where to store the updated gradient value.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (GPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        cudanet.dot(deltas._tensor, inputs.transpose()._tensor, out._tensor)

    #def fprop_conv

    #def bprop_conv

    #def update_conv

    #def fprop_pool

    #def bprop_pool

    #def fprop_cmrnorm

    #def bprop_cmrnorm

    #def fprop_lcnnorm

    #def bprop_lcnnorm

    #def fprop_cmpool

    #def bprop_cmpool

    #def update_cmpool

    #def ada_update

    def sync_stream(self):
        cudanet.sync_stream()

    def set_weights(self, dev_weights, host_weights):
        """
        sets the GPUTensor dev_weights to the values in host_weights
        """
        dev_weights[:] = GPUTensor(numpy.array(host_weights, 'float32'))
