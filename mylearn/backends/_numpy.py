"""
Wraps numpy ndarray and operations into our backend interface.
"""

import logging
import math
import numpy as np

from mylearn.backends.backend import Backend

logger = logging.getLogger(__name__)


class Numpy(Backend):
    """
    Sets up a numpy based backend for matrix ops.
    """

    class Tensor(Backend.Tensor):
        """
        Simple wrapped numpy ndarray tensor

        Arguments:
            obj (numpy.ndarray): the actual data values
            dtype (numpy.ndtype): underlying data type of the elements
        """
        _tensor = None

        def __init__(self, obj, dtype=None):
            if type(obj) != np.ndarray:
                self._tensor = np.array(obj, dtype)
            else:
                self._tensor = obj
            self.shape = self._tensor.shape

        def __str__(self):
            return str(self._tensor)

        def __getitem__(self, key):
            return Numpy.Tensor(self._tensor[key])

        def __setitem__(self, key, value):
            self._tensor[key] = value
        # TODO: __delitem__ implementation

        def __float__(self):
            return float(self._tensor)

        def __neg__(self):
            return Numpy.Tensor(- self._tensor)

        def __lt__(self, other):
            if isinstance(other, Numpy.Tensor):
                return self._tensor < other._tensor
            else:
                return self._tensor < other

        def __le__(self, other):
            if isinstance(other, Numpy.Tensor):
                return self._tensor <= other._tensor
            else:
                return self._tensor <= other

        def __eq__(self, other):
            if isinstance(other, Numpy.Tensor):
                return self._tensor == other._tensor
            else:
                return self._tensor == other

        def __ne__(self, other):
            if isinstance(other, Numpy.Tensor):
                return self._tensor != other._tensor
            else:
                return self._tensor != other

        def __gt__(self, other):
            if isinstance(other, Numpy.Tensor):
                return self._tensor > other._tensor
            else:
                return self._tensor > other

        def __ge__(self, other):
            if isinstance(other, Numpy.Tensor):
                return self._tensor >= other._tensor
            else:
                return self._tensor >= other

        def __add__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(self._tensor + other._tensor)
            else:
                return Numpy.Tensor(self._tensor + other)

        def __radd__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(other._tensor + self._tensor)
            else:
                return Numpy.Tensor(other + self._tensor)

        def __sub__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(self._tensor - other._tensor)
            else:
                return Numpy.Tensor(self._tensor - other)

        def __rsub__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(other._tensor - self._tensor)
            else:
                return Numpy.Tensor(other - self._tensor)

        def __mul__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(self._tensor * other._tensor)
            else:
                return Numpy.Tensor(self._tensor * other)

        def __rmul__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(other._tensor * self._tensor)
            else:
                return Numpy.Tensor(other * self._tensor)

        def __div__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(self._tensor / other._tensor)
            else:
                return Numpy.Tensor(self._tensor / other)

        def __rdiv__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(other._tensor / self._tensor)
            else:
                return Numpy.Tensor(other / self._tensor)

        def __pow__(self, other, modulo=None):
            # TODO: determine how ternary modulo needs to be handled
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(self._tensor ** other._tensor)
            else:
                return Numpy.Tensor(self._tensor ** other)

        def __rpow__(self, other):
            if isinstance(other, Numpy.Tensor):
                return Numpy.Tensor(other._tensor ** self._tensor)
            else:
                return Numpy.Tensor(other ** self._tensor)

        def copy(self):
            return Numpy.Tensor(np.copy(self._tensor))

        def raw(self):
            return self._tensor

        def T(self):
            return Numpy.Tensor(self._tensor.T)

        def transpose(self):
            return Numpy.Tensor(self._tensor.T)

        def reshape(self, shape):
            return Numpy.Tensor(self._tensor.reshape(shape))

        def argmax(self, axis):
            return Numpy.Tensor(self._tensor.argmax(axis))

        def get(self, indices, axis):
            if type(indices) == Numpy.Tensor:
                indices = indices._tensor
            return Numpy.Tensor(self._tensor.take(indices, axis))

        def take(self, indices, axis=None):
            if type(indices) == Numpy.Tensor:
                indices = indices._tensor
            return Numpy.Tensor(self._tensor.take(indices, axis))

        def get_slice(self, start, end, axis):
            """
            Return a view made of consecutive rows/columns.
            """
            return Numpy.Tensor(self._tensor.take(range(start, end), axis))

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
            if type(indices) == Numpy.Tensor:
                indices = indices._tensor
            if axis == 0:
                self._tensor[indices, :] = obj._tensor
            elif axis == 1:
                self._tensor[:, indices] = obj._tensor
            else:
                raise NotImplementedError()

        def set_slice(self, obj, start, end, axis):
            """
            Copy the input tensor into consecutive rows/columns.
            """
            if axis == 0:
                self._tensor[range(start, end), :] = obj._tensor
            elif axis == 1:
                self._tensor[:, range(start, end)] = obj._tensor
            else:
                raise NotImplementedError()

        def set_elems(self, obj, indices, axis):
            if type(indices) == Numpy.Tensor:
                indices = indices._tensor
            if axis == 0:
                self._tensor[range(self._tensor.shape[0]),
                             indices] = obj._tensor
            elif axis == 1:
                self._tensor[indices,
                             range(self._tensor.shape[1])] = obj._tensor
            else:
                raise NotImplementedError()

        def add(self, obj):
            self._tensor += obj._tensor

        def sub(self, obj):
            self._tensor -= obj._tensor

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.rng_init()

    def zeros(self, shape, dtype=float):
        return self.Tensor(np.zeros(shape, dtype))

    def array(self, obj):
        return self.Tensor(np.array(obj))

    def rng_init(self):
        if 'rng_seed' in self.__dict__:
            np.random.seed(self.rng_seed)
        else:
            raise AttributeError("rng_seed not specified in config")

    def uniform(self, low=0.0, high=1.0, size=1):
        """
        Uniform random number generation of samples in range [low, high).
        Returns a new Tensor of size size filled with these random numbers.
        """
        return self.Tensor(np.random.uniform(low, high, size))

    def normal(self, loc=0.0, scale=1.0, size=1):
        """
        Gaussian/Normal random number generation of samples centered around
        mean loc, and with standard deviation scale.
        Returns a new Tensor of size size filled with these random numbers.
        """
        return self.Tensor(np.random.normal(loc, scale, size))

    def append_bias(self, x):
        """
        Adds a bias column to Tensor x, returning a new Tensor.
        """
        return self.Tensor(np.concatenate((x._tensor,
                                          np.ones((x.shape[0], 1))),
                                          axis=1))

    def copy(self, a):
        return self.Tensor(np.copy(a))

    def argmax(self, x, axis=None):
        return self.Tensor(np.argmax(x._tensor, axis))

    def dot(self, a, b):
        return self.Tensor(np.dot(a._tensor, b._tensor))

    def sum(self, obj):
        return obj._tensor.sum()

    def mean(self, x, axis=None, dtype=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.mean(x._tensor, axis, dtype, out, keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return self.Tensor(res)

    def min(self, x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.min(x._tensor, axis, out, keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return self.Tensor(res)

    def max(self, x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.max(x._tensor, axis, out, keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return self.Tensor(res)

    def squish(self, obj, n):
        assert obj.shape[1] % n == 0
        return obj.reshape((obj.shape[0] * n, obj.shape[1] / n))

    def not_equal(self, x, y):
        return self.Tensor(np.not_equal(x._tensor, y._tensor))

    def nonzero(self, x):
        return self.Tensor(np.nonzero(x._tensor)[1])

    def logistic(self, x):
        return self.Tensor(1.0 / (1.0 + np.exp(-x._tensor)))

    def logistic_prime(self, x):
        y = self.logistic(x)
        return y * (1.0 - y)

    def pseudo_logistic(self, x):
        return 1.0 / (1.0 + 2 ** -x)

    def pseudo_logistic_prime(self, z):
        y = self.pseudo_logistic(z)
        return math.log(2) * y * (1.0 - y)

    def tanh(self, x):
        y = np.exp(-2 * x)
        return (1.0 - y) / (1.0 + y)

    def tanh_prime(self, x):
        y = self.tanh(x)
        return 1.0 - y * y

    def rectlin(self, x):
        xc = x.copy()
        xc[xc < 0] = 0
        return xc

    def rectlin_prime(self, x):
        xc = x.copy()
        xc[xc < 0] = 0
        xc[xc != 0] = 1
        return xc

    def noact(self, x):
        return x

    def noact_prime(self, x):
        return self.Tensor(np.ones(x.shape))

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
            weights = self.uniform(low, high, size)
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
            weights = self.normal(loc, scale, size)
        elif weight_params['type'] == 'node_normalized':
            # initialization is as discussed in Glorot2010
            scale = 1.0
            if 'scale' in weight_params:
                scale = weight_params['scale']
            logger.info('generating %s node_normalized(%0.2f) weights.' %
                        (str(size), scale))
            node_norm = scale * math.sqrt(6.0 / sum(size))
            weights = self.uniform(-node_norm, node_norm, size)
        else:
            raise AttributeError("invalid weight_params specified")
        if 'bias_init' in weight_params:
            # per append_bias() bias weights are in the last column
            logger.info('separately initializing bias weights to %0.2f' %
                        weight_params['bias_init'])
            weights[:, -1] = weight_params['bias_init']

        return weights

    def get_momentum_coef(self, epoch, momentum_params):
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
        return self.Tensor(np.mean(-targets * np.log(outputs) -
                           (1 - targets) * np.log(1 - outputs)))

    def cross_entropy_de(self, outputs, targets):
        outputs = outputs._tensor
        targets = targets._tensor
        return self.Tensor((outputs - targets) / (outputs * (1.0 - outputs)))

    def sse(self, outputs, targets):
        """ Sum of squared errors """
        return 0.5 * np.sum((outputs - targets) ** 2)

    def sse_de(self, outputs, targets):
        """ Derivative of SSE with respect to the output """
        return (outputs - targets)
