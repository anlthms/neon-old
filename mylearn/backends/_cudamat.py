"""
A wrapped cudamat GPU based backend.
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
    A cudamat based backend for matrix ops.
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
    def dot(a, b):
        return CudamatTensor(cudamat.dot(a._tensor, b._tensor))

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

    @staticmethod
    def exp(x):
        target = cudamat.empty(x.shape)
        cudamat.exp(x._tensor, target)
        return CudamatTensor(target)

    @staticmethod
    def log(x):
        target = cudamat.empty(x.shape)
        cudamat.log(x._tensor, target)
        return CudamatTensor(target)

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
    Wrapped CUDAMatrix tensor
    """
    _tensor = None

    def __init__(self, obj, dtype=None):
        if type(obj) == cudamat.CUDAMatrix:
            self._tensor = obj
        else:
            # CUDAMatrix only supports ndarrays with exactly 2 dimensions
            if isinstance(obj, (float, int, str, list, tuple)):
                obj = numpy.array(obj)
            if type(obj) == numpy.ndarray:
                while obj.ndim < 2:
                    obj = obj.reshape(obj.shape + (1, ))
                if obj.ndim != 2:
                    raise ValueError("CUDAMatrix only supports 2-D"
                                     "matrices.  You specifed %d-D" %
                                     obj.ndim)
            logger.debug('Copying to GPU')
            self._tensor = cudamat.CUDAMatrix(obj)
        self.shape = self._tensor.shape

    def __str__(self):
        return str(self._tensor.asarray())

    def __getstate__(self):
        """
        Defines what and how we go about serializing an instance of this class.
        """
        return self._tensor.asarray()

    def __setstate__(self, state):
        """
        Defines how we go about deserializing into an instance of this class.
        """
        self.__init__(state)

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
            # raise TooSlowToImplementError("column idx too complex")
            res = self.get(_slice, dim)
        return res

    def __getitem__(self, key):
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
                            start, stop, stride = (key[1].indices(
                                                   self.shape[1]))
                            self._tensor.set_col_slice(start, stop, value)
                        elif isinstance(key[1], int):
                            self._tensor.set_col_slice(key[1], key[1] + 1,
                                                       value)
                        else:
                            raise TooSlowToImplementError("arbitrary "
                                                          "indexing")
                    elif isinstance(key[1], slice):
                        start_1, stop_1, stride_1 = (key[1].indices(
                                                     self.shape[1]))
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
                        start_1, stop_1, stride_1 = (key[1].indices(
                                                     self.shape[1]))
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
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.add(other._tensor, target)
        else:
            self._tensor.add(other, target)
        return CudamatTensor(target)

    def __radd__(self, other):
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.add(other._tensor, target)
        else:
            self._tensor.add(other, target)
        return CudamatTensor(target)

    def __iadd__(self, other):
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
            self._tensor.add(other._tensor, target)
        else:
            self._tensor.add(other, target)
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
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.divide(other._tensor, target)
        else:
            self._tensor.divide(other, target)
        return CudamatTensor(target)

    def __rdiv__(self, other):
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

    def __pow__(self, other, modulo=None):
        target = cudamat.empty(self.shape)
        if isinstance(other, CudamatTensor):
            self._tensor.pow(other._tensor, target)
        else:
            self._tensor.pow(other, target)
        return CudamatTensor(target)

    def __rpow__(self, other):
        target = cudamat.empty(self.shape)
        if isinstance(other, (float, int)):
            other = CudamatTensor(other)
        if isinstance(other, CudamatTensor):
            other._tensor.pow(self._tensor, target)
        elif isinstance(other, cudamat.CUDAMatrix):
            other.pow(self._tensor, target)
        else:
            return NotImplemented
        return CudamatTensor(target)

    def __ipow__(self, other):
        if isinstance(other, CudamatTensor):
            self._tensor.pow(other._tensor)
        else:
            self._tensor.pow(other)
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
        # we only support contiguous indices at the moment because this
        # is all cudamat supports efficiently.
        if len(indices) == 0:
            return self
        if (indices[-1] - indices[0] == len(indices) - 1 and
            (len(indices) <= 1 or all(x < y for x, y in zip(indices,
                                      indices[1:])))):
            if axis == 0:
                return CudamatTensor(self._tensor.get_row_slice(
                                     indices[0], indices[-1] + 1))
            elif axis == 1:
                return CudamatTensor(self._tensor.get_col_slice(
                                     indices[0], indices[-1] + 1))
            elif axis is None:
                # we might be able to do this by first doing a reshape?
                raise TooSlowToImplementError("need to first reshape")
        else:
            raise TooSlowToImplementError("CUDAMatrix can't do arbitrary"
                                          " indexing efficiently")

    def get(self, indices, axis=None):
        # FIXME: This routine is terribly expensive! Should return a view
        # instead of a newly allocated matrix.
        if type(indices) == int:
            indices = [indices]
        elif type(indices) == CudamatTensor:
            raise NotImplementedError()
        if axis == 0 or axis is None:
            mat = cudamat.empty((len(indices), self._tensor.shape[1]))
            dst_ind = 0
            for src_ind in indices:
                src_ind = int(src_ind)
                row = self._tensor.get_row_slice(src_ind, src_ind + 1)
                mat.set_row_slice(dst_ind, dst_ind + 1, row)
                dst_ind += 1
        elif axis == 1:
            mat = cudamat.empty((self._tensor.shape[0], len(indices)))
            dst_ind = 0
            for src_ind in indices:
                src_ind = int(src_ind)
                col = self._tensor.get_col_slice(src_ind, src_ind + 1)
                mat.set_col_slice(dst_ind, dst_ind + 1, col)
                dst_ind += 1
        else:
            raise NotImplementedError()
        return CudamatTensor(mat)

    def add(self, obj):
        self._tensor.add(obj._tensor)

    def sub(self, obj):
        self._tensor.subtract(obj._tensor)

    def sum(self):
        result = self._tensor.sum(axis=0).sum(axis=1)
        logger.debug('Copying to host')
        result.copy_to_host()
        return result.numpy_array[0][0]

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


class TransposedCudamatTensor(CudamatTensor):
    """
    Transposed CUDAMatrix tensor
    """

    def __init__(self, obj, transposed):
        assert type(obj) == cudamat.CUDAMatrix
        self._tensor = transposed
        self.shape = (obj.shape[1], obj.shape[0])
