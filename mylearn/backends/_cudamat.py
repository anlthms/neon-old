"""
A wrapped cudamat GPU based backend.
"""

import logging
import numpy
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

        def take(self, start, end, axis):
            if axis == 0:
                return Cudamat.Tensor(self._tensor.get_row_slice(start, end))
            if axis == 1:
                return Cudamat.Tensor(self._tensor.get_col_slice(start, end))
            raise NotImplementedError()  # if axis == None

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

    def rng_init(self):
        if 'rng_seed' in self.__dict__:
            numpy.random.seed(self.rng_seed)
        else:
            raise AttributeError("rng_seed not specified in config")

    def uniform(self, low=0.0, high=1.0, size=1):
        return self.Tensor(numpy.random.uniform(low, high, size))

    def normal(self, loc=0.0, scale=1.0, size=1):
        return self.Tensor(numpy.random.normal(loc, scale, size))

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

    def not_equal(self, x, y):
        raise NotImplementedError()

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
        raise NotImplementedError()

    def rectlin_prime(self, x):
        raise NotImplementedError()

    def noact(self, x):
        return x

    def noact_prime(self, x):
        return self.Tensor(numpy.ones(x.shape))

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
        if weight_params['type'] == 'uniform':
            low = 0.0
            high = 1.0
            if 'low' in weight_params:
                low = weight_params['low']
            if 'high' in weight_params:
                high = weight_params['high']
            logger.info('generating %s uniform(%0.2f, %0.2f) weights.' %
                        (str(size), low, high))
            return self.uniform(low, high, size)
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
            return self.normal(loc, scale, size)

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
