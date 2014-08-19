"""
A `cudamat <https://github.com/cudamat/cudamat>`_ GPU based backend.
"""

import logging
import numpy
import math
import cudamat

from mylearn.backends.backend import Backend, Tensor

logger = logging.getLogger(__name__)


class TooSlowToImplementError(Exception):

    """
    Used to indicate types of operations that would take too long to run.
    """
    pass


class Cudamat(Backend):

    """
    A `cudamat <https://github/com/cudamat/cudamat>`_ based backend for matrix
    operations.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.rng_init()
        cudamat.cublas_init()

    def zeros(self, shape, dtype=float):
        return CudamatTensor(cudamat.CUDAMatrix(
            numpy.zeros(shape, dtype=numpy.float32)))

    @staticmethod
    def array(obj):
        ndarray = numpy.array(obj, dtype=numpy.float32)
        if ndarray.ndim == 1:
            ndarray = ndarray.reshape((1, ndarray.shape[0]))
        return CudamatTensor(ndarray)

    @staticmethod
    def wrap(obj):
        return CudamatTensor(obj)

    def rng_init(self):
        if 'rng_seed' in self.__dict__:
            numpy.random.seed(self.rng_seed)
        else:
            raise AttributeError("rng_seed not specified in config")

    def uniform(self, low=0.0, high=1.0, size=1):
        seq = numpy.random.uniform(low, high, size)
        return CudamatTensor(numpy.array(seq, dtype=numpy.float32))

    def normal(self, loc=0.0, scale=1.0, size=1):
        seq = numpy.random.normal(loc, scale, size)
        return CudamatTensor(numpy.array(seq, dtype=numpy.float32))

    def append_bias(self, x):
        """
        Adds a bias column to CudamatTensor x, returning a new CudamatTensor.
        """
        result = cudamat.empty((x.shape[0], x.shape[1] + 1))
        result.set_col_slice(0, x.shape[1], x._tensor)
        result.set_col_slice(x.shape[1], (x.shape[1] + 1),
                             cudamat.CUDAMatrix.ones.slice(0, x.shape[0]))
        return CudamatTensor(result)

    @staticmethod
    def copy(a):
        assert type(a) == CudamatTensor
        return a.copy()

    @staticmethod
    def argmax(x, axis=None):
        return CudamatTensor(x._tensor.argmax(axis))

    @staticmethod
    def dot(a, b, out):
        cudamat.dot(a._tensor, b._tensor, out._tensor)

    @staticmethod
    def add(a, b, out):
        a._tensor.add(b._tensor, out._tensor)

    @staticmethod
    def subtract(a, b, out):
        if type(a._tensor) != cudamat.CUDAMatrix:
            b._tensor.subtract(a._tensor, out._tensor)
            out._tensor.mult(-1.0, out._tensor)
        else:
            a._tensor.subtract(b._tensor, out._tensor)

    @staticmethod
    def multiply(a, b, out):
        a._tensor.mult(b._tensor, target=out._tensor)

    @staticmethod
    def divide(a, b, out):
        a._tensor.divide(b._tensor, out._tensor)

    @staticmethod
    def reciprocal(a, out):
        a._tensor.reciprocal(out._tensor)

    @staticmethod
    def greater(a, b, out):
        a._tensor.greater_than(b._tensor, out._tensor)

    @staticmethod
    def exp(x, out):
        cudamat.exp(x._tensor, out._tensor)

    @staticmethod
    def log(x, out):
        cudamat.log(x._tensor, out._tensor)

    @staticmethod
    def logistic(x, out):
        cudamat.sigmoid(x._tensor, out._tensor)

    @staticmethod
    def sum(x):
        if x is None:
            return float('NaN')
        return x.sum()

    @staticmethod
    def mean(x):
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
            res = cudamat.min(x._tensor, axis)
        else:
            res = cudamat.min(x._tensor, axis, out)

        return CudamatTensor(res)

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
            res = cudamat.max(x._tensor, axis)
        else:
            res = cudamat.max(x._tensor, axis, out)

        return CudamatTensor(res)

    def sqrt(self, x, out):
        res = cudamat.sqrt(x._tensor, out._tensor)
        return CudamatTensor(res)

    def squish(self, obj, n):
        assert obj.shape[1] % n == 0
        return obj.reshape((obj.shape[0] * n, obj.shape[1] / n))

    def not_equal(self, x, y):
        res = x._tensor.copy()
        res.equals(y._tensor)
        res.equals(0)
        return CudamatTensor(res)

    def nonzero(self, x):
        raise NotImplementedError()

    def gen_weights(self, size, weight_params):
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

        return CudamatTensor(numpy.array(weights, numpy.float32))

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


class CudamatTensor(Tensor):

    """
    Simple wrapped `cudamat.CUDAMatrix` tensor

    Arguments:
        obj (numpy.ndarray): the actual data values (will be converted
                             to 2-d row matrix).  Python built-in types like
                             lists and tuples are also supported.
        dtype (None, optional): underlying data type of the elements.
                                Not needed/used for this backend.
    """
    _tensor = None

    def __init__(self, obj, dtype=None):
        if type(obj) == cudamat.CUDAMatrix:
            self._tensor = obj
            self.shape = self._tensor.shape
        else:
            if type(obj) == list:
                obj = numpy.array(obj)

            if type(obj) == numpy.ndarray:
                # CUDAMatrix only supports ndarrays with exactly 2 dimensions
                # (though the elements can be tuples/lists to create arbitrary n
                # dimensions)
                #if isinstance(obj, (float, int, str, list, tuple)):
                #    obj = numpy.array(obj)
                while obj.ndim < 2:
                    obj = obj.reshape(obj.shape + (1, ))
                if obj.ndim != 2:
                    raise ValueError("CUDAMatrix only supports 2-D"
                                     "matrices.  You specifed %d-D" %
                                     obj.ndim)
                logger.debug('Copying to GPU')
                self._tensor = cudamat.CUDAMatrix(obj)
                self.shape = self._tensor.shape
            else:
                self._tensor = obj

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
                           `cudamat.CUDAMatrix` tensor
        """
        return self._tensor.asarray()

    def __setstate__(self, state):
        """
        Defines how we go about deserializing into an instance of this class.

        Arguments:
            state (numpy.ndarray): Serialized representation of the underlying
                                   `cudamat.CUDAMatrix` tensor to be unpacked.
        """
        self.__init__(state)
        if not hasattr(cudamat.CUDAMatrix, 'ones'):
            cudamat.cublas_init()

    def _slice_dim(self, _slice, dim=0):
        """
        Helper that actually performs a slice along the dimension passed.

        Arguments:
            _slice (int or slice): actual slice object specifying indices
            dim (int): dimension number. 0 is for rows, 1 for columns, etc.

        Returns:
            CudamatTensor: view or new sliced copy

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
            res = CudamatTensor(fn(start, stop))
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
            CudamatTensor: view or new sliced copy

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
            value (CudamatTensor, int, float): The values to assign at each
                                               key position.  Must be scalar
                                               or if a CudamatTensor, must
                                               have the right shape.

        Returns:
            CudamatTensor: update view of this tensor

        Raises:
            IndexError: if invalid number of dimensions specified in key.
            NotImplementedError: if invalid value type passed.
            TooSlowToImplementError: if arbitrarily indexed key passed.
        """
        if isinstance(value, CudamatTensor):
            value = value._tensor
        elif not isinstance(value, (int, float)):
            raise NotImplementedError("can only assign Cudamat tensors or "
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
            raise IndexError("Cudamat only supports 2-D fancy indexing")

    def __delitem__(self, key):
        raise ValueError("cannot delete array elements")

    def __float__(self):
        raise NotImplementedError()

    def __neg__(self):
        return -1 * self

    def __lt__(self, other):
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.less_than(other._tensor, target)
        else:
            self._tensor.less_than(other, target)
        return CudamatTensor(target)

    def __le__(self, other):
        # call __lt__ and __eq__ and iterate?
        raise NotImplementedError()

    def __eq__(self, other):
        if other is None:
            return False
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.equals(other._tensor, target)
        else:
            self._tensor.equals(other, target)
        return CudamatTensor(target)

    def __ne__(self, other):
        # go through results of __eq__ and negate
        raise NotImplementedError()

    def __gt__(self, other):
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.greater_than(other._tensor, target)
        else:
            self._tensor.greater_than(other, target)
        return CudamatTensor(target)

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
            CudamatTensor: containing the element-wise sum values.
        """

        if self.shape == other.shape:
            target = cudamat.empty(self.shape)
            if isinstance(other, CudamatTensor):
                self._tensor.add(other._tensor, target)
            else:
                self._tensor.add(other, target)
            return CudamatTensor(target)
        else:
            if other.shape[1] == 1:  # [Nx1] vector
                ones = cudamat.empty((self.shape[0], 1))
                ones.assign(1)
                # outer product repmat (probably quite inefficient)
                other = CudamatTensor(cudamat.dot(ones, other._tensor.T))
            else:  # [1xN] vector
                ones = cudamat.empty((self.shape[0], 1))
                ones.assign(1)
                other = CudamatTensor(cudamat.dot(ones, other._tensor))
            target = cudamat.empty(self.shape)
            if isinstance(other, CudamatTensor):
                self._tensor.add(other._tensor, target)
            else:
                self._tensor.add(other, target)
            return CudamatTensor(target)

    def __radd__(self, other):
        """
        Perform element-wise addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same
                            dimensions as this Tensor, or be broadcastable
                            as such.

        Returns:
            CudamatTensor: containing the element-wise sum values.
        """
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.add(other._tensor, target)
        else:
            self._tensor.add(other, target)
        return CudamatTensor(target)

    def __iadd__(self, other):
        """
        Perform element-wise in-place addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same
                            dimensions as this Tensor, or be broadcastable
                            as such.

        Returns:
            CudamatTensor: updated view of this Tensor
        """
        if isinstance(other, CudamatTensor):
            self._tensor.add(other._tensor)
        else:
            self._tensor.add(other)
        return self

    def __sub__(self, other):
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.subtract(other._tensor, target)
        else:
            self._tensor.subtract(other, target)
        return CudamatTensor(target)

    def __rsub__(self, other):
        target = cudamat.empty(self.shape)
        self._tensor.mult(-1.0, target)
        if isinstance(other, CudamatTensor):
            target.add(other._tensor)
        else:
            target.add(other)
        return CudamatTensor(target)

    def __isub__(self, other):
        if isinstance(other, CudamatTensor):
            self._tensor.subtract(other._tensor)
        else:
            self._tensor.subtract(other)
        return self

    def __mul__(self, other):
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.mult(other._tensor, target)
        else:
            self._tensor.mult(other, target)
        return CudamatTensor(target)

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        if isinstance(other, CudamatTensor):
            self._tensor.mult(other._tensor)
        else:
            self._tensor.mult(other)
        return self

    def __div__(self, other):
        # python2 floor rounded division
        return self.__truediv__(other)

    def __truediv__(self, other):
        # python3 fractional division
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.divide(other._tensor, target)
        else:
            self._tensor.divide(other, target)
        return CudamatTensor(target)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rtruediv__(self, other):
        target = cudamat.empty(self.shape)
        if isinstance(other, (float, int)):
            other = CudamatTensor(other * numpy.ones(self.shape))
        if isinstance(other, CudamatTensor):
            other._tensor.divide(self._tensor, target)
        elif isinstance(other, cudamat.CUDAMatrix):
            other.divide(self._tensor, target)
        else:
            return NotImplemented
        return CudamatTensor(target)

    def __idiv__(self, other):
        if isinstance(other, CudamatTensor):
            self._tensor.divide(other._tensor)
        else:
            self._tensor.divide(other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, CudamatTensor):
            self._tensor.divide(other._tensor)
        else:
            self._tensor.divide(other)
        return self

    def __pow__(self, other, modulo=None):
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            cudamat.pow(self._tensor, other._tensor, target)
        else:
            cudamat.pow(self._tensor, other, target)
        return CudamatTensor(target)

    def __rpow__(self, other):
        target = cudamat.empty(self.shape)
        if isinstance(other, (float, int)):
            other = CudamatTensor(other)
        if isinstance(other, CudamatTensor):
            cudamat.pow(other._tensor, self._tensor, target)
        elif isinstance(other, cudamat.CUDAMatrix):
            cudamat.pow(other, self._tensor, target)
        else:
            return NotImplemented
        return CudamatTensor(target)

    def __ipow__(self, other):
        if isinstance(other, CudamatTensor):
            cudamat.pow(self._tensor, other._tensor)
        else:
            cudamat.pow(self._tensor, other)
        return self

    def copy(self):
        return CudamatTensor(self._tensor.copy())

    def raw(self):
        self._tensor.copy_to_host()
        return self._tensor.numpy_array

    def T(self):
        # CUDAMatrix.T is a transposed view.
        return TransposedCudamatTensor(self._tensor, self._tensor.T)

    def transpose(self):
        # CUDAMatrix.transpose() returns a transposed copy.
        return CudamatTensor(self._tensor.transpose())

    def reshape(self, shape):
        return CudamatTensor(self._tensor.reshape(shape))

    def argmax(self, axis):
        return CudamatTensor(self._tensor.argmax(axis))

    def take(self, indices, axis=None):
        """
        Take returns a subset of a tensor specified by indices.
        Urs modified this to be consistent with numpy, where vectors
        get flipped to always be rows.
        """
        # we only support contiguous indices at the moment because this
        # is all cudamat supports efficiently.
        if isinstance(indices, int):
            indices = [indices, ]  # cudamat only supports 2D matrix
            if self._tensor.shape[0] == 1:
                axis = 1
                # fix the axis if we are dealing with a vector. This is a hack
                # and should be done differently.
        if len(indices) == 0:
            return self
        if (indices[-1] - indices[0] == len(indices) - 1):
            if axis == 0:
                return CudamatTensor(self._tensor.get_row_slice(indices[0],
                                                                indices[-1] +
                                                                1))
            elif axis == 1:
                return CudamatTensor(self._tensor.get_col_slice(indices[0],
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
        Sum elements of a CudamatTensor. If axis is None, all elements are
        summed and a numpy scalar returned. If axis is 1 or 2, sum along that
        axis and return a CudamatTensor.
        """
        if axis is None:
            result = self._tensor.sum(axis=0).sum(axis=1)
            logger.debug('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]
        else:
            result = self._tensor.sum(axis=axis)
            logger.debug('major change in functionality of sum')
            return CudamatTensor(result)

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
        target = cudamat.empty(self.shape)
        cudamat.log(self._tensor, target)
        return CudamatTensor(target)

    def exp(self):
        target = cudamat.empty(self.shape)
        cudamat.exp(self._tensor, target)
        return CudamatTensor(target)


class TransposedCudamatTensor(CudamatTensor):

    """
    Transposed CUDAMatrix tensor
    """

    def __init__(self, obj, transposed):
        assert type(obj) == cudamat.CUDAMatrix
        self._tensor = transposed
        self.shape = (obj.shape[1], obj.shape[0])
