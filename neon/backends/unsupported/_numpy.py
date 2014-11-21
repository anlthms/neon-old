# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Wraps :mod:`numpy` ndarray and operations into our backend interface.
"""

import logging
import math
import numpy as np

from neon.backends.backend import Backend, Tensor

logger = logging.getLogger(__name__)


class NumpyTensor(Tensor):
    """
    Simple wrapped `numpy.ndarray` tensor

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        dtype (numpy.ndtype, optional): underlying data type of the elements.
                                        If None will use float32.

    See also:
        Numpy, NumpyTensor64
    """
    _tensor = None

    def __init__(self, obj, dtype=None):
        if dtype is None:
            dtype = np.float32
        if type(obj) != np.ndarray:
            self._tensor = np.array(obj, dtype)
        elif obj.dtype != dtype:
            self._tensor = obj.astype(dtype)
        else:
            self._tensor = obj
        self.shape = self._tensor.shape
        self.dtype = dtype

    def __str__(self):
        """
        Display a suitable representation of this Tensor.

        Returns:
            str: the representation.
        """
        return str(self._tensor)

    def __repr__(self):
        return ("%s(%s)" %
                (self.__class__.__name__, str(self)))

    def _clean(self, val):
        """
        Replaces any NumpyTensor indices with `numpy` arrays.

        Arguments:
            val (int, array_like, NumpyTensor): the items to index by.

        Returns:
            int, array_like, NumpyTensor: Transformed val
        """
        if isinstance(val, tuple):
            val = tuple(x._tensor if isinstance(x, self.__class__) else x
                        for x in val)
        if isinstance(val, self.__class__):
            val = val._tensor
        return val

    def __getitem__(self, key):
        return self.__class__(self._tensor[self._clean(key)],
                              dtype=self._tensor.dtype)

    def __setitem__(self, key, value):
        self._tensor[self._clean(key)] = self._clean(value)

    def __delitem__(self, key):
        raise ValueError("cannot delete array elements")

    def __float__(self):
        return float(self._tensor)

    def __neg__(self):
        return self.__class__(- self._tensor,
                              dtype=self._tensor.dtype)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor < other._tensor)
        else:
            return self.__class__(self._tensor < other)

    def __le__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor <= other._tensor)
        else:
            return self.__class__(self._tensor <= other)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor == other._tensor)
        else:
            return self.__class__(self._tensor == other)

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor != other._tensor)
        else:
            return self.__class__(self._tensor != other)

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor > other._tensor)
        else:
            return self.__class__(self._tensor > other)

    def __ge__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor >= other._tensor)
        else:
            return self.__class__(self._tensor >= other)

    def __add__(self, other):
        """
        Perform element-wise addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same dimensions
                            as this Tensor, or be broadcastable as such.

        Returns:
            self.__class__: containing the element-wise sum values.
        """
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor + other._tensor)
        else:
            return self.__class__(self._tensor + other)

    def __radd__(self, other):
        """
        Perform element-wise addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same dimensions
                            as this Tensor, or be broadcastable as such.

        Returns:
            self.__class__: containing the element-wise sum values.
        """
        if isinstance(other, self.__class__):
            return self.__class__(other._tensor + self._tensor)
        else:
            return self.__class__(other + self._tensor)

    def __iadd__(self, other):
        """
        Perform element-wise in-place addition with the items in other.

        Arguments:
            other (Tensor): The Tensor to add.  Must have the same dimensions
                            as this Tensor, or be broadcastable as such.

        Returns:
            self.__class__: containing the element-wise sum values.
        """
        if isinstance(other, self.__class__):
            self._tensor += other._tensor
        else:
            self._tensor += other
        return self

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor - other._tensor)
        else:
            return self.__class__(self._tensor - other)

    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(other._tensor - self._tensor)
        else:
            return self.__class__(other - self._tensor)

    def __isub__(self, other):
        if isinstance(other, self.__class__):
            self._tensor -= other._tensor
        else:
            self._tensor -= other
        return self

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor * other._tensor)
        else:
            return self.__class__(self._tensor * other)

    def __rmul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(other._tensor * self._tensor)
        else:
            return self.__class__(other * self._tensor)

    def __imul__(self, other):
        if isinstance(other, self.__class__):
            self._tensor *= other._tensor
        else:
            self._tensor *= other
        return self

    def __div__(self, other):
        # python2 floor rounded division
        return self.__truediv__(other)

    def __truediv__(self, other):
        # python3 fractional division
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor / other._tensor)
        else:
            return self.__class__(self._tensor / other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(other._tensor / self._tensor)
        else:
            return self.__class__(other / self._tensor)

    def __idiv__(self, other):
        if isinstance(other, self.__class__):
            self._tensor /= other._tensor
        else:
            self._tensor /= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, self.__class__):
            self._tensor /= other._tensor
        else:
            self._tensor /= other
        return self

    def __pow__(self, other, modulo=None):
        # TODO: determine how ternary modulo needs to be handled
        if isinstance(other, self.__class__):
            return self.__class__(self._tensor ** other._tensor)
        else:
            return self.__class__(self._tensor ** other)

    def __rpow__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(other._tensor ** self._tensor)
        else:
            return self.__class__(other ** self._tensor)

    def __ipow__(self, other):
        if isinstance(other, self.__class__):
            self._tensor **= other._tensor
        else:
            self._tensor **= other
        return self

    def copy(self):
        return self.__class__(np.copy(self._tensor),
                              dtype=self._tensor.dtype)

    def raw(self):
        return self._tensor

    def T(self):  # flake8: noqa
        return self.__class__(self._tensor.T,
                              dtype=self._tensor.dtype)

    def transpose(self):
        return self.__class__(self._tensor.T,
                              dtype=self._tensor.dtype)

    def reshape(self, shape):
        # TODO: Some layer code (ex. PoolingLayer) currently depends
        # on squish/reshape always returning a view of the existing
        # data, but numpy.reshape does not guarantee this.  We should remove
        # reliance on this dependency.
        return self.__class__(self._tensor.reshape(shape),
                              dtype=self._tensor.dtype)

    def argmax(self, axis):
        return self.__class__(self._tensor.argmax(axis))

    def take(self, indices, axis=None):
        if type(indices) == self.__class__:
            indices = indices._tensor
        return self.__class__(self._tensor.take(indices, axis),
                              self._tensor.dtype)

    def add(self, obj):
        self._tensor += obj._tensor

    def sub(self, obj):
        self._tensor -= obj._tensor

    def norm(self, axis):
        return self.__class__(np.sqrt((self._tensor * self._tensor).sum(axis)))

    def repeat(self, repeats, axis):
        return self.__class__(self._tensor.repeat(repeats, axis))

    def log(self):
        return self.__class__(np.log(self._tensor))

    def exp(self):
        return self.__class__(np.exp(self._tensor))

    def mean(self, axis=None, dtype=np.float32, out=None):
        res = np.mean(self._tensor, axis, dtype, out)
        if axis is None:
            return res
        else:
            return self.__class__(res)

    def sum(self, axis=None, dtype=np.float32, out=None):
        res = np.sum(self._tensor, axis, dtype, out)
        if axis is None:
            return res
        else:
            return self.__class__(res)

    def sumsq(self, axis=None, dtype=np.float32, out=None):
        res = np.sum(self._tensor * self._tensor, axis, dtype, out)
        if axis is None:
            return res
        else:
            return self.__class__(res)

    def get_minor_slice(self, start, end):
        return self.__class__(self[start:end]._tensor)

    def set_minor_slice(self, start, end, data):
        self[start:end] = data

    def get_major_slice(self, start, end):
        return self.__class__(self[:, start:end]._tensor)

    def set_major_slice(self, start, end, data):
        self[:, start:end] = data

    def major_axis(self):
        return 0

    def minor_axis(self):
        return 1


class Numpy64Tensor(NumpyTensor):
    """
    Simple wrapped `numpy.ndarray` tensor, defaults to 64-bit elements.

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        dtype (numpy.ndtype, optional): underlying data type of the elements.
                                        If None will use `numpy.float64`.

    See also:
        Numpy64, NumpyTensor
    """
    _tensor = None

    def __init__(self, obj, dtype=None):
        if dtype is None:
            dtype = np.float64
        if type(obj) != np.ndarray:
            self._tensor = np.array(obj, dtype)
        else:
            self._tensor = obj
            if self._tensor.dtype != np.float64:
                self._tensor = self._tensor.astype(np.float64)
        self.shape = self._tensor.shape
        self.dtype = dtype


class Numpy(Backend):
    """
    Sets up a :mod:`numpy` based backend for matrix ops.  By default, we use
    32-bit element data types for any arrays constructed.

    Attributes:
        default_dtype (dtype): default element data type.  We assume 32-bit
                               float
        epsilon (float): the unit roundoff for the elements underlying this
                         tensor.
    See also:
        Numpy64, NumpyTensor
    """
    default_dtype = np.float32
    epsilon = np.finfo(np.float32).eps
    tensor_cls = NumpyTensor

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.err_init()
        self.rng_init()

    def default_dtype_if_missing(self, in_dtype):
        if in_dtype is None:
            in_dtype = self.default_dtype
        return in_dtype

    def empty(self, shape, dtype=None):
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.empty(shape, dtype), dtype)

    def zeros(self, shape, dtype=None):
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.zeros(shape, dtype), dtype)

    def alloc(self, nrows, ncols, dtype=None):
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.zeros((nrows, ncols), dtype), dtype)

    def ones(self, shape, dtype=None):
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.ones(shape, dtype), dtype)

    def array(self, obj, dtype=None):
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.array(obj, dtype), dtype)

    def wrap(self, obj, dtype=None):
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(obj, dtype)

    def clip(self, a, a_min, a_max, out=None):
        if out is None:
            out = self._tensor_cls(np.empty_like(a._tensor))
        np.clip(a._tensor, a_min, a_max, out._tensor)
        return out

    def err_init(self):
        # support numpy.seterr settings:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html
        if 'seterr_handling' in self.__dict__:
            logger.info("Updating numpy.seterr settings: %s" %
                        str(self.seterr_handling))
            np.seterr(**self.seterr_handling)

    def rng_init(self):
        seed = None
        if 'rng_seed' in self.__dict__:
            seed = self.rng_seed
            logger.info("Seeding random number generator with: %s" % str(seed))
        np.random.seed(seed)

    def uniform(self, low=0.0, high=1.0, size=1, dtype=None):
        """
        Uniform random number sample generation.

        Arguments:
            low (float, optional): Minimal sample value that can be returned.
                                   Defaults to 0.0
            high (float, optional): Maximal sample value.  Open ended range so
                                    maximal value slightly less.  Defaults to
                                    1.0
            size (array_like or int, optional): Shape of generated samples

        Returns:
            Tensor: Of specified size filled with these random numbers.
        """
        return self.tensor_cls(np.random.uniform(low, high, size), dtype)

    def normal(self, loc=0.0, scale=1.0, size=1, dtype=None):
        """
        Gaussian/Normal random number sample generation

        Arguments:
            loc (float, optional): Where to center distribution.  Defaults
                                   to 0.0
            scale (float, optional): Standard deviaion.  Defaults to 1.0
            size (array_like or int, optional): Shape of generated samples

        Returns:
            Tensor: Of specified size filled with these random numbers.
        """
        return self.tensor_cls(np.random.normal(loc, scale, size), dtype)

    def append_bias(self, x, dtype=np.float32):
        """
        Adds a bias column of ones to NumpyTensor x,
        returning a new NumpyTensor.
        """
        return self.tensor_cls(np.concatenate((x._tensor,
                                               np.ones((x.shape[0], 1),
                                                        dtype)),
                                              axis=1), dtype)

    def copy(self, a):
        return self.tensor_cls(np.copy(a))

    def argmax(self, x, axis=None):
        return self.tensor_cls(np.argmax(x._tensor, axis))

    def dot(self, a, b, out):
        np.dot(a._tensor, b._tensor, out._tensor)

    def add(self, a, b, out):
        np.add(a._tensor, b._tensor, out._tensor)

    def subtract(self, a, b, out):
        np.subtract(a._tensor, b._tensor, out._tensor)

    def multiply(self, a, b, out):
        np.multiply(a._tensor, b._tensor, out._tensor)

    def divide(self, a, b, out):
        np.divide(a._tensor, b._tensor, out._tensor)

    def reciprocal(self, a, out):
        np.divide(1.0, a._tensor, out._tensor)

    def greater(self, a, b, out):
        np.greater(a._tensor, b._tensor, out._tensor)

    def xcov(self, a, b, out):
        a0 = a._tensor - a._tensor.mean(0, keepdims=True)
        b0 = b._tensor - b._tensor.mean(0, keepdims=True)
        np.dot(a0.T, b0, out._tensor)
        Numpy.divide(out, Numpy.wrap(a.shape[0]), out=out)

    def mean_norm(self, a, axis, out):
        if (axis=-1 or axis=None):
            out._tensor = a._tensor - a._tensor.mean()
        else:
            out._tensor = a._tensor - a._tensor.mean(axis, keepdims=True)

    def exp(self, x, out):
        np.exp(x._tensor, out=out._tensor)

    def log(self, x, out):
        np.log(x._tensor, out=out._tensor)

    def logistic(self, x, out):
        self.multiply(x, self.wrap(-1.0), out=out)
        self.exp(out, out=out)
        self.add(out, self.wrap(1.0), out=out)
        self.reciprocal(out, out=out)

    def clear(self, x):
        x._tensor[:] = 0

    def fill(self, x, val):
        x._tensor.fill(val)

    def sum(self, obj):
        return obj._tensor.sum()

    def mean(self, x, axis=None, dtype=np.float32, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.mean(x._tensor, axis, dtype, out, keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return self.tensor_cls(res)

    def min(self, x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.min(x._tensor, axis, out, keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return self.tensor_cls(res)

    def max(self, x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.max(x._tensor, axis, out, keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return self.tensor_cls(res)

    def fabs(self, x, out=None):
        if out is not None:
            res = np.fabs(x._tensor, out._tensor)
        else:
            res = np.fabs(x._tensor)
        return self.tensor_cls(res)

    def sqrt(self, x, out):
        res = np.sqrt(x._tensor, out._tensor)
        return self.tensor_cls(res)

    def square(self, x, out):
        np.multiply(x._tensor, x._tensor, out._tensor)

    def cube(self, x, out):
        np.multiply(x._tensor, x._tensor, out._tensor)
        np.multiply(out._tensor, x._tensor, out._tensor)

    def squish(self, obj, n):
        """ reshape a tensor by increasing the first dimensions by factor n,
        and shrinking the the second dimension by factor n."""
        assert obj.shape[1] % n == 0
        return obj.reshape((obj.shape[0] * n, obj.shape[1] / n))

    def not_equal(self, x, y):
        return self.tensor_cls(np.not_equal(x._tensor, y._tensor))

    def fprop_conv(self, weights, inputs, outputs, links, ifmshape, ofmshape,
                   ofmlocs, padding, stride, nifm, ngroups, prodbuf):
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            rflinks = links[dst]
            self.dot(inputs.take(rflinks, axis=1), weights, out=prodbuf)
            outputs[:, ofmlocs[dst]] = prodbuf

    def bprop_conv(self, weights, error, berror, links, ifmshape, ofmshape,
                   ofmlocs, padding, stride, nifm, ngroups, bpropbuf):
        self.fill(berror, 0.0)
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            self.dot(error.take(ofmlocs[dst], axis=1), weights.T(), bpropbuf)
            rflinks = links[dst]
            self.add(bpropbuf, berror.take(rflinks, axis=1), out=bpropbuf)
            berror[:, rflinks] = bpropbuf

    def update_conv(self, weights, inputs, error, updates, links, ifmshape,
                    ofmshape, ofmlocs, padding, stride, nifm, ngroups, fwidth,
                    updatebuf):
        self.fill(updates, 0.0)
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = links[dst]
            eslice = error.take(ofmlocs[dst], axis=1)
            self.dot(inputs.take(rflinks, axis=1).T(), eslice,
                     out=updatebuf)
            updates.add(updatebuf)

    def fprop_mpool(self, inputs, outputs, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm, maxinds):
        rinputs = self.squish(inputs, nfm)
        routputs = self.squish(outputs, nfm)
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            # For this output unit, get the corresponding receptive fields
            # within all input feature maps.
            rf = rinputs.take(links[dst], axis=1)
            # Save the index of the maximum value within the receptive fields.
            maxinds[:, dst] = rf.argmax(axis=1)
            # Set the pre-activations to the maximum value.
            maxvals = rf[range(rf.shape[0]), maxinds[:, dst]]
            routputs[:, dst] = maxvals

    def bprop_mpool(self, inputs, outputs, error, berror, links, ifmshape,
                    ofmshape, fshape, padding, stride, nfm, maxinds):
        self.fill(berror, 0.0)
        rberror = self.squish(berror, nfm)
        rerror = self.squish(error, nfm)
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            rflinks = links[dst]
            inds = rflinks.take(maxinds[:, dst], axis=0)
            rberror[range(rberror.shape[0]), inds] += rerror[:, dst]

    def fprop_apool(self, inputs, outputs, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm):
        rinputs = self.squish(inputs, nfm)
        routputs = self.squish(outputs, nfm)
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            rf = rinputs.take(links[dst], axis=1)
            routputs[:, dst] = rf.mean(axis=1)

    def bprop_apool(self, outputs, error, berror, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm):
        self.fill(berror, 0.0)
        error /= fshape[0] * fshape[1]
        rberror = self.squish(berror, nfm)
        rerror = self.squish(error, nfm)
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            rberror[:, links[dst]] += rerror[:, dst:(dst + 1)]

    def fprop_l2pool(self, inputs, outputs, links, ifmshape, ofmshape,
                    fshape, padding, stride, nfm):
        rinputs = self.squish(inputs, nfm)
        routputs = self.squish(outputs, nfm)
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            rf = rinputs.take(links[dst], axis=1)
            routputs[:, dst] = rf.norm(axis=1)

    def bprop_l2pool(self, inputs, outputs, error, berror, links, ifmshape,
                    ofmshape, fshape, padding, stride, nfm, prodbuf):
        rinputs = self.squish(inputs, nfm)
        routputs = self.squish(outputs, nfm)
        rberror = self.squish(berror, nfm)
        rerror = self.squish(error, nfm)
        self.fill(berror, 0.0)
        for dst in xrange(ofmshape[0] * ofmshape[1]):
            inds = links[dst]
            rf = rinputs.take(inds, axis=1)
            denom = routputs[:, dst:(dst + 1)].copy()
            # If the L2 norm is zero, the entire receptive field must be
            # zeros. In that case, we set the L2 norm to 1 before using
            # it to normalize the receptive field.
            denom[denom.raw() == 0] = 1
            self.divide(rf, denom, out=rf)
            self.multiply(
                rerror[:, dst:(dst + 1)].repeat(fshape[0] * fshape[1], axis=1),
                rf, out=prodbuf)
            rberror[:, inds] += prodbuf

    def fprop_fc_dot(self, inputs, weights, out):
        np.dot(inputs._tensor, weights.T()._tensor, out._tensor)

    def bprop_fc_dot(self, deltas, weights, out):
        np.dot(deltas._tensor, weights._tensor, out._tensor)

    def update_fc_dot(self, deltas, inputs, out):
        np.dot(deltas.T()._tensor, inputs._tensor, out._tensor)

    def format(self, raw):
        return self.array(raw)

    def gen_weights(self, size, weight_params, dtype=None):
        weights = None
        if 'dtype' in weight_params:
            dtype = weight_params['dtype']
        if weight_params['type'] == 'uniform':
            low = 0.0
            high = 1.0
            if 'low' in weight_params:
                low = weight_params['low']
            if 'high' in weight_params:
                high = weight_params['high']
            logger.info('generating %s uniform(%0.2f, %0.2f) weights.' %
                        (str(size), low, high))
            weights = self.uniform(low, high, size, dtype)
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
            weights = self.normal(loc, scale, size, dtype)
        elif weight_params['type'] == 'node_normalized':
            # initialization is as discussed in Glorot2010
            scale = 1.0
            if 'scale' in weight_params:
                scale = weight_params['scale']
            logger.info('generating %s node_normalized(%0.2f) weights.' %
                        (str(size), scale))
            node_norm = scale * math.sqrt(6.0 / sum(size))
            weights = self.uniform(-node_norm, node_norm, size, dtype)
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


class Numpy64(Numpy):
    """
    Sets up a :mod:`numpy` based backend for matrix ops.  By default, we use
    64-bit element data types for any arrays constructed.

    See also:
        Numpy, Numpy64Tensor
    """
    default_dtype = np.float64
    epsilon = np.finfo(np.float64).eps
    tensor_cls = Numpy64Tensor

    def gen_weights(self, size, weight_params, dtype=None):
        if dtype is None:
            dtype=np.float64
        return super(Numpy64, self).gen_weights(size, weight_params, dtype)

    def append_bias(self, x, dtype=np.float64):
        return Numpy64Tensor(np.concatenate((x._tensor,
                                            np.ones((x.shape[0], 1), dtype)),
                                            axis=1), dtype)

    def clip(self, a, a_min, a_max, out=None):
        if out is None:
            out = Numpy64Tensor(np.empty_like(a._tensor))
        np.clip(a._tensor, a_min, a_max, out._tensor)
        return out
