"""
A wrapped cudamat GPU based backend.
"""

import logging
import numpy
import math
import cudamat

from mylearn.backends.backend import Backend

logger = logging.getLogger(__name__)


class Cudamat(Backend):
    """
    A cudamat based backend for matrix ops.
    """

    class Tensor(Backend.Tensor):
        """
        Wrapped CUDAMatrix tensor
        """
        _tensor = None

        def __init__(self, obj, dtype=None):
            if type(obj) == cudamat.CUDAMatrix:
                self._tensor = obj
            else:
                logger.info('Copying to GPU')
                self._tensor = cudamat.CUDAMatrix(obj)
            self.shape = self._tensor.shape

        def __str__(self):
            return str(self._tensor)

        def __getitem__(self, key):
            raise NotImplementedError()

        def __setitem__(self, key, value):
            raise NotImplementedError()

        def __float__(self):
            raise NotImplementedError()

        def __neg__(self):
            raise NotImplementedError()

        def __lt__(self, other):
            raise NotImplementedError()

        def __le__(self, other):
            raise NotImplementedError()

        def __eq__(self, other):
            if other is None:
                return False
            raise NotImplementedError()

        def __ne__(self, other):
            raise NotImplementedError()

        def __gt__(self, other):
            raise NotImplementedError()

        def __ge__(self, other):
            raise NotImplementedError()

        def __add__(self, other):
            target = cudamat.empty(self.shape)
            if isinstance(other, Cudamat.Tensor):
                self._tensor.add(other._tensor, target)
            else:
                self._tensor.add(other, target)
            return Cudamat.Tensor(target)

        def __radd__(self, other):
            raise NotImplementedError()

        def __sub__(self, other):
            target = cudamat.empty(self.shape)
            if isinstance(other, Cudamat.Tensor):
                self._tensor.subtract(other._tensor, target)
            else:
                self._tensor.subtract(other, target)
            return Cudamat.Tensor(target)

        def __rsub__(self, other):
            target = cudamat.empty(self.shape)
            self._tensor.mult(-1.0, target)
            if isinstance(other, Cudamat.Tensor):
                self._tensor.add(other._tensor, target)
            else:
                self._tensor.add(other, target)
            return Cudamat.Tensor(target)

        def __mul__(self, other):
            target = cudamat.empty(self.shape)
            if isinstance(other, Cudamat.Tensor):
                self._tensor.mult(other._tensor, target)
            else:
                self._tensor.mult(other, target)
            return Cudamat.Tensor(target)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __div__(self, other):
            target = cudamat.empty(self.shape)
            if isinstance(other, Cudamat.Tensor):
                self._tensor.divide(other._tensor, target)
            else:
                self._tensor.divide(other, target)
            return Cudamat.Tensor(target)

        def __rdiv__(self, other):
            raise NotImplementedError()

        def __pow__(self, other, modulo=None):
            raise NotImplementedError()

        def __rpow__(self, other):
            raise NotImplementedError()

        def copy(self):
            return Cudamat.Tensor(self._tensor.copy())

        def raw(self):
            self._tensor.copy_to_host()
            return self._tensor.numpy_array

        def T(self):
            # CUDAMatrix.T is a transposed view.
            return Cudamat.TransposedTensor(self._tensor, self._tensor.T)

        def transpose(self):
            # CUDAMatrix.transpose() returns a transposed copy.
            return Cudamat.Tensor(self._tensor.transpose())

        def reshape(self, shape):
            return Cudamat.Tensor(self._tensor.reshape(shape))

        def argmax(self, axis):
            return Cudamat.Tensor(self._tensor.argmax(axis))

        def get(self, indices, axis):
            # FIXME: This routine is terribly expensive! Should return a view instead
            # of a newly allocated matrix.
            if type(indices) == int:
                indices = [indices]
            elif type(indices) == Cudamat.Tensor:
                raise NotImplementedError()
            if axis == 0:
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
            return Cudamat.Tensor(mat)

        def get_slice(self, start, end, axis):
            """
            Return a view made of consecutive rows/columns.
            """
            if axis == 0:
                return Cudamat.Tensor(self._tensor.get_row_slice(start, end))
            if axis == 1:
                return Cudamat.Tensor(self._tensor.get_col_slice(start, end))
            raise NotImplementedError()

        def get_elems(self, indices, axis):
            assert type(indices) == Numpy.Tensor
            if axis == 0:
                return Numpy.Tensor(self._tensor[range(self._tensor.shape[0]),
                                                 indices._tensor])
            if axis == 1:
                return Numpy.Tensor(self._tensor[indices._tensor,
                                                 range(self._tensor.shape[1])])
            raise NotImplementedError()

        def set(self, obj, indices, axis):
            """
            This is the opposite of get(). Copy the input tensor into the
            rows/columns specified by indices.
            """
            if type(indices) == int:
                indices = [indices]
            elif type(indices) == Cudamat.Tensor:
                raise NotImplementedError()
            tensor = obj._tensor
            src_ind = 0
            for dst_ind in indices:
                dst_ind = int(dst_ind)
                if axis == 0:
                    self._tensor.set_row_slice(dst_ind, dst_ind + 1,
                            tensor.get_row_slice(src_ind, src_ind + 1))
                elif axis == 1:
                    self._tensor.set_col_slice(dst_ind, dst_ind + 1,
                            tensor.get_col_slice(src_ind, src_ind + 1))
                else:
                    raise NotImplementedError()
                src_ind += 1

        def set_slice(self, obj, start, end, axis):
            """
            Copy the input tensor into consecutive rows/columns.
            """
            if axis == 0:
                self._tensor.set_row_slice(start, end, obj._tensor)
            elif axis == 1:
                self._tensor.set_col_slice(start, end, obj._tensor)
            else:
                raise NotImplementedError()

        def add(self, obj):
            self._tensor.add(obj._tensor)

        def sub(self, obj):
            self._tensor.subtract(obj._tensor)

        def sum(self):
            result = self._tensor.sum(axis=0).sum(axis=1)
            logger.info('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]

        def mean(self):
            result = self._tensor.mean(axis=0).mean(axis=1)
            logger.info('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]

        def min(self):
            result = self._tensor.min(axis=0).min(axis=1)
            logger.info('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]

        def max(self):
            result = self._tensor.max(axis=0).max(axis=1)
            logger.info('Copying to host')
            result.copy_to_host()
            return result.numpy_array[0][0]

    class TransposedTensor(Tensor):
        """
        Transposed CUDAMatrix tensor
        """

        def __init__(self, obj, transposed):
            assert type(obj) == cudamat.CUDAMatrix
            self._tensor = transposed
            self.shape = (obj.shape[1], obj.shape[0])

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.rng_init()
        cudamat.cublas_init()

    def zeros(self, shape, dtype=float):
        return self.Tensor(cudamat.CUDAMatrix(numpy.zeros(shape,
                                                          dtype=numpy.float32)))

    def array(self, obj):
        ndarray = numpy.array(obj, dtype=numpy.float32)
        if ndarray.ndim == 1:
            ndarray = ndarray.reshape((1, ndarray.shape[0]))
        return self.Tensor(ndarray)

    def rng_init(self):
        if 'rng_seed' in self.__dict__:
            numpy.random.seed(self.rng_seed)
        else:
            raise AttributeError("rng_seed not specified in config")

    def uniform(self, low=0.0, high=1.0, size=1):
        seq = numpy.random.uniform(low, high, size)
        return self.Tensor(numpy.array(seq, dtype=numpy.float32))

    def normal(self, loc=0.0, scale=1.0, size=1):
        seq = numpy.random.normal(loc, scale, size)
        return self.Tensor(numpy.array(seq, dtype=numpy.float32))

    def append_bias(self, x):
        """
        Adds a bias column to Tensor x, returning a new Tensor.
        """
        result = cudamat.empty((x.shape[0], x.shape[1] + 1))
        result.set_col_slice(0, x.shape[1], x._tensor)
        result.set_col_slice(x.shape[1], (x.shape[1] + 1),
                             cudamat.CUDAMatrix.ones.slice(0, x.shape[0]))
        return self.Tensor(result)

    def copy(self, a):
        assert type(a) == Cudamat.Tensor
        return a.copy()

    def argmax(self, x, axis=None):
        return self.Tensor(x._tensor.argmax(axis))

    def dot(self, a, b):
        return self.Tensor(cudamat.dot(a._tensor, b._tensor))

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
            logger.info('Copying to host')
            res.copy_to_host()
            return res.numpy_array[0][0]

        if out is None:
            res = cudamat.min(x._tensor, axis)
        else:
            res = cudamat.min(x._tensor, axis, out)

        return self.Tensor(res)

    def max(self, x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        if axis is None and not keepdims:
            assert out is None
            res = x._tensor.max(axis=0).max(axis=1)
            logger.info('Copying to host')
            res.copy_to_host()
            return res.numpy_array[0][0]

        if out is None:
            res = cudamat.max(x._tensor, axis)
        else:
            res = cudamat.max(x._tensor, axis, out)

        return self.Tensor(res)

    def squish(self, obj, n):
        assert obj.shape[1] % n == 0
        return obj.reshape((obj.shape[0] * n, obj.shape[1] / n))

    def not_equal(self, x, y):
        res = x._tensor.copy()
        res.equals(y._tensor)
        res.equals(0)
        return self.Tensor(res)

    def nonzero(self, x):
        raise NotImplementedError()

    def logistic(self, x):
        target = cudamat.empty(x.shape)
        cudamat.sigmoid(x._tensor, target)
        return self.Tensor(target)

    def logistic_prime(self, x):
        y = self.logistic(x)._tensor
        result = y.copy()
        result.mult(-1.0)
        result.add(1.0)
        result.mult(y)
        return self.Tensor(result)

    def pseudo_logistic(self, x):
        raise NotImplementedError()

    def pseudo_logistic_prime(self, z):
        raise NotImplementedError()

    def tanh(self, x):
        raise NotImplementedError()

    def tanh_prime(self, x):
        raise NotImplementedError()

    def rectlin(self, x):
        xc = x._tensor.copy()
        mask = cudamat.empty(xc.shape)
        xc.greater_than(0, mask)
        xc.mult(mask)
        return self.Tensor(xc)

    def rectlin_prime(self, x):
        xc = x._tensor.copy()
        xc.greater_than(0)
        return self.Tensor(xc)

    def noact(self, x):
        return x

    def noact_prime(self, x):
        return self.Tensor(numpy.ones(x.shape, dtype=numpy.float32))

    def get_derivative(self, func):
        if func == self.logistic:
            return self.logistic_prime
        if func == self.pseudo_logistic:
            return self.pseudo_logistic_prime
        if func == self.tanh:
            return self.tanh_prime
        if func == self.rectlin:
            return self.rectlin_prime
        if func == self.noact:
            return self.noact_prime
        if func == self.cross_entropy:
            return self.cross_entropy_de
        if func == self.sse:
            return self.sse_de

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

        return self.Tensor(numpy.array(weights, numpy.float32))

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

    def cross_entropy(self, outputs, targets):
        outputs = outputs._tensor
        targets = targets._tensor

        negative_targets = targets.copy()
        negative_targets.mult(-1.0)
        term1 = outputs.copy()
        cudamat.log(term1)
        term1.mult(negative_targets)

        term2 = outputs.copy()
        term2.mult(-1.0)
        term2.add(1.0, target=term2)
        cudamat.log(term2)

        reverse_targets = negative_targets
        reverse_targets.add(1.0)
        term2.mult(reverse_targets)

        diff = term1
        diff.subtract(term2)
        diff_tensor = self.Tensor(diff)
        return diff_tensor.mean()

    def cross_entropy_de(self, outputs, targets):
        outputs = outputs._tensor
        targets = targets._tensor

        result = outputs.copy()
        result.subtract(targets)

        denom = outputs.copy()
        denom.mult(-1.0)
        denom.add(1.0)
        denom.mult(outputs)
        result.divide(denom)
        return self.Tensor(result)

    def sse(self, outputs, targets):
        """ Sum of squared errors """
        diff = outputs - targets
        return 0.5 * (diff * diff).sum()

    def sse_de(self, outputs, targets):
        """ Derivative of SSE with respect to the output """
        return (outputs - targets)
