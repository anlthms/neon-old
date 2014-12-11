# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Our CPU based backend interface and tensor data structure.  Our implementation
wraps :mod:`numpy` ndarray and related operations
"""

import logging
import math
import numpy as np
from neon.util.compat import MPI_INSTALLED, range

from neon.backends.backend import Backend, Tensor

if MPI_INSTALLED:
    from mpi4py import MPI

logger = logging.getLogger(__name__)


class CPUTensor(Tensor):

    """
    Our basic n-dimensional array data structure that resides in host memory,
    and is meant to be manipulated on the CPU.  wrapped `numpy.ndarray` tensor.

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        dtype (numpy.ndtype, optional): underlying data type of the elements.
                                        If None will use float32.

    See also:
        CPU
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
        Replaces any CPUTensor indices with `numpy` arrays.

        Arguments:
            val (int, array_like, CPUTensor): the items to index by.

        Returns:
            int, array_like, CPUTensor: Transformed val
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

    def asnumpyarray(self):
        return self._tensor

    def __float__(self):
        return float(self._tensor)

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

    def transpose(self):
        return self.__class__(self._tensor.transpose(),
                              dtype=self._tensor.dtype)

    def reshape(self, shape):
        return self.__class__(self._tensor.reshape(shape),
                              dtype=self._tensor.dtype)

    def argmax(self, axis):
        return self.__class__(self._tensor.argmax(axis))

    def take(self, indices, axis=None):
        if type(indices) == self.__class__:
            indices = indices._tensor
        return self.__class__(self._tensor.take(indices, axis),
                              self._tensor.dtype)

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


class CPU(Backend):

    """
    Sets up a :mod:`numpy` based backend for matrix ops.  By default, we use
    32-bit element data types for any arrays constructed.

    Attributes:
        default_dtype (dtype): default element data type.  We assume 32-bit
                               float
        epsilon (float): the unit roundoff for the elements underlying this
                         tensor.
    See also:
        CPUTensor
    """
    default_dtype = np.float32
    epsilon = np.finfo(np.float32).eps
    tensor_cls = CPUTensor

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.err_init()
        self.rng_init()

    def default_dtype_if_missing(self, in_dtype):
        if in_dtype is None:
            in_dtype = self.default_dtype
        return in_dtype

    def empty(self, shape, dtype=None):
        """
        Instantiate a new instance of the CPUTensor class without initializing
        individual element values.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value (np.float32
                                     unless overridden).

        Returns:
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.empty(shape, dtype), dtype)

    def array(self, obj, dtype=None):
        """
        Instantiate a new instance of the CPUTensor class setting each element
        value to what is specified in obj.

        Arguments:
            obj (numpy.ndarray): The data structure containing element values
                                 spread across a number of dimensions.  Python
                                 built-in types like ints and lists are
                                 supported.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value (np.float32
                                     unless overridden).

        Returns:
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.array(obj, dtype), dtype)

    def zeros(self, shape, dtype=None):
        """
        Instantiate a new instance of the CPUTensor class setting each element
        value to 0.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value (np.float32
                                     unless overridden).

        Returns:
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.zeros(shape, dtype), dtype)

    def ones(self, shape, dtype=None):
        """
        Instantiate a new instance of the CPUTensor class setting each element
        value to 1.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specified we
                                     use default_dtype value (np.float32
                                     unless overridden).

        Returns:
            CPUTensor: newly created data structure reference
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.ones(shape, dtype), dtype)

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
            logger.info("Updating numpy.seterr settings: %s",
                        str(self.seterr_handling))
            np.seterr(**self.seterr_handling)

    def rng_init(self):
        seed = None
        if 'rng_seed' in self.__dict__:
            seed = self.rng_seed
            logger.info("Seeding random number generator with: %s", str(seed))
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

    def fill_uniform_thresh(self, a, keepthresh=0.5, dtype=None):
        """
        Uniform random number sample generation.

        Arguments:
            a (dtype): CPUTensor to fill with zeros or ones based on whether
                       sample from uniform distribution is < keepthresh
            keepthresh (float, optional): Minimal sample value that can be
                                          returned. Defaults to 0.5
        Returns:
            Tensor: Of specified size filled with these random numbers.
        """
        a._tensor[:] = np.array(
            np.random.uniform(size=a._tensor.shape) < keepthresh,
            dtype=a._tensor.dtype)
        a._tensor[:] = a._tensor[:] / keepthresh

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
        Adds a bias row of ones to CPUTensor x,
        returning a new CPUTensor.
        """
        return self.tensor_cls(np.concatenate((x._tensor,
                                               np.ones((1, x.shape[1]),
                                                       dtype)),
                                              axis=0), dtype)

    def copy(self, a):
        return self.tensor_cls(np.copy(a))

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

    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor): left-hand side operand.
            right (CPUTensor): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        return np.equal(left._tensor, right._tensor, out._tensor)

    def not_equal(self, left, right, out):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor): left-hand side operand.
            right (CPUTensor): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        return np.not_equal(left._tensor, right._tensor, out._tensor)

    def greater(self, left, right, out):
        """
        Performs element-wise greater than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor): left-hand side operand.
            right (CPUTensor): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        return np.greater(left._tensor, right._tensor, out._tensor)

    def greater_equal(self, left, right, out):
        """
        Performs element-wise greater than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor): left-hand side operand.
            right (CPUTensor): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        return np.greater_equal(left._tensor, right._tensor, out._tensor)

    def less(self, left, right, out):
        """
        Performs element-wise less than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor): left-hand side operand.
            right (CPUTensor): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        return np.less(left._tensor, right._tensor, out._tensor)

    def less_equal(self, left, right, out):
        """
        Performs element-wise less than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (CPUTensor): left-hand side operand.
            right (CPUTensor): right-hand side operand.
            out (CPUTensor): where the result will be stored.

        Returns:
            CPUTensor: reference to out
        """
        return np.less_equal(left._tensor, right._tensor, out._tensor)

    def norm(self, tsr, order=None, axis=None, out=None):
        """
        Calculates and returns the vector p-norms of the CPUTensor along the
        specified axis.  The p-norm is defined on vector A as
        :math:`||A||_p = \sum_i(|A_i|^p)^{1/p}`.

        Arguments:
            tsr (CPUTensor): the CPUTensor on which to find the norms
            order (int): The order or p upon which the norm is calculated.
                         Valid values include:
                         None, inf, -inf, 0, 1, -1, 2, -2, ...
            axis (int): The axis along which to compute vector norms.
            out (CPUTensor, optional): where to write the results to.  Must be
                                       of the expected result shape.  If not
                                       specified, a new buffer is created and
                                       returned.

        Returns:
            CPUTensor: p-norm of tsr along the specified axis.

        See Also:
            `numpy.linalg.norm`
        """
        if not isinstance(axis, int):
            raise AttributeError("invalid axis value: %s", axis)
        if order == float('Inf'):
            res = np.max(np.abs(tsr._tensor), axis)
        elif order == float('-Inf'):
            res = np.min(np.abs(tsr._tensor), axis)
        elif order == 0:
            res = np.sum(tsr._tensor != 0, axis)
        else:
            res = np.sum(np.abs(tsr._tensor) ** order, axis) ** (1.0 / order)
        if out is None:
            out = self.array(res)
        else:
            out._tensor = res
            out.shape = res.shape
        return out

    def xcov(self, a, b, out):
        a0 = a._tensor - a._tensor.mean(1, keepdims=True)
        b0 = b._tensor - b._tensor.mean(1, keepdims=True)
        np.dot(a0, b0.T, out._tensor)
        self.divide(out, self.wrap(a.shape[0]), out=out)

    def mean_norm(self, a, axis, out):
        if (axis == -1 or not axis):
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

    def tanh(self, x, out):
        np.exp(-2.0 * x._tensor, out=out._tensor)
        np.divide(1. - out._tensor, 1. + out._tensor, out=out._tensor)

    def rectlin(self, x, out):
        self.greater(x, self.wrap(0), out=out)
        self.multiply(x, out, out=out)

    def rectlin_derivative(self, x, out):
        self.greater(x, self.wrap(0), out=out)

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

    def argmin(self, tsr, axis, out):
        """
        Calculates the indices of the minimal element value along the specified
        axis.  If multiple elements contain the minimum, only the elements of
        the first are returned.

        Arguments:
            tsr (CPUTensor): The CPUTensor on which to find the minimum indices
            axis (int): The dimension along which to find the minimum.  If set
                        to None, find the overall minimum index of a flattened
                        representation of tsr.
            out (CPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape
        """
        out._tensor[:] = np.argmin(tsr._tensor, axis)

    def argmax(self, tsr, axis, out):
        """
        Calculates the indices of the maximal element value along the specified
        axis.  If multiple elements contain the maximum, only the elements of
        the first are returned.

        Arguments:
            tsr (CPUTensor): The CPUTensor on which to find the maximum indices
            axis (int): The dimension along which to find the maximum.  If set
                        to None, find the overall maximum index of a flattened
                        representation of tsr.
            out (CPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape
        """
        out._tensor[:] = np.argmax(tsr._tensor, axis)

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

    def power(self, x, a, out):
        np.power(x._tensor, a._tensor, out._tensor)

    # Not part of the API - can be moved to a utility class.
    def hstack_maps(self, obj, nfm):
        """
        Stack the feature maps horizontally.
        """
        assert obj.shape[0] % nfm == 0
        return self.tensor_cls(np.hstack(np.vsplit(obj._tensor, nfm)))

    # Not part of the API - can be moved to a utility class.
    def vstack_maps(self, obj, nfm):
        """
        Stack the feature maps vertically.
        """
        assert obj.shape[1] % nfm == 0
        return self.tensor_cls(np.vstack(np.hsplit(obj._tensor, nfm)))

    def softmax(self, x, out):
        x._tensor.max(axis=0, out=out._tensor[0])
        np.subtract(x._tensor, x._tensor.max(axis=0, keepdims=True),
                    out._tensor)
        np.exp(out._tensor, out._tensor)
        # This uses some temporary storage, but might be ok?
        np.divide(out._tensor, np.sum(out._tensor, axis=0, keepdims=True),
                  out._tensor)

    def softmax_gradient(self, y, err, out):
        a = np.einsum('ij,ji->i', err._tensor.T, y._tensor)
        np.subtract(err._tensor, a[np.newaxis], out._tensor)
        np.multiply(out._tensor, y._tensor, out._tensor)

    def fprop_conv(self, weights, inputs, outputs, links, ifmshape, ofmshape,
                   ofmlocs, padding, stride, nifm, ngroups, prodbuf):
        for dst in range(ofmshape[0] * ofmshape[1]):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            rflinks = links[dst]
            self.dot(weights.transpose(), inputs.take(rflinks, axis=0),
                     out=prodbuf)
            outputs[ofmlocs[dst]] = prodbuf

    def bprop_conv(self, weights, error, berror, links, ifmshape, ofmshape,
                   ofmlocs, padding, stride, nifm, ngroups, bpropbuf):
        self.fill(berror, 0.0)
        for dst in range(ofmshape[0] * ofmshape[1]):
            self.dot(weights, error.take(ofmlocs[dst], axis=0), bpropbuf)
            rflinks = links[dst]
            self.add(bpropbuf, berror.take(rflinks, axis=0), out=bpropbuf)
            berror[rflinks] = bpropbuf

    def update_conv(self, weights, inputs, error, updates, links, ifmshape,
                    ofmshape, ofmlocs, padding, stride, nifm, ngroups, fwidth,
                    updatebuf):
        self.fill(updates, 0.0)
        for dst in range(ofmshape[0] * ofmshape[1]):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = links[dst]
            eslice = error.take(ofmlocs[dst], axis=0)
            self.dot(inputs.take(rflinks, axis=0), eslice.transpose(),
                     out=updatebuf)
            self.add(updates, updatebuf, out=updates)

    def fprop_mpool(self, inputs, outputs, outputsbuf, links,
                    ifmshape, ofmshape, fshape, padding, stride, nfm, maxinds):
        rinputs = self.hstack_maps(inputs, nfm)
        for dst in range(ofmshape[0] * ofmshape[1]):
            # For this output unit, get the corresponding receptive fields
            # within all input feature maps.
            rf = rinputs.take(links[dst], axis=0)
            # Save the index of the maximum value within the receptive fields.
            maxinds[dst] = rf.argmax(axis=0)
            # Set the pre-activations to the maximum value.
            maxvals = rf[maxinds[dst], range(rf.shape[1])]
            outputsbuf[dst] = maxvals
        outputs[:] = self.vstack_maps(outputsbuf, nfm)

    def bprop_mpool(self, inputs, outputs, error, berror, berrorbuf, links,
                    ifmshape, ofmshape, fshape, padding, stride, nfm, maxinds):
        self.fill(berrorbuf, 0.0)
        rerror = self.hstack_maps(error, nfm)
        for dst in range(ofmshape[0] * ofmshape[1]):
            rflinks = links[dst]
            inds = rflinks.take(maxinds[dst], axis=0)
            berrorbuf[inds, range(berrorbuf.shape[1])] += rerror[dst]
        berror[:] = self.vstack_maps(berrorbuf, nfm)

    # Alternate implementation of max pooling fprop. To be deleted.
    def fprop_mpool2(self, inputs, outputs, links, ifmshape, ofmshape,
                     fshape, padding, stride, nfm, maxinds):
        ifmsize = ifmshape[0] * ifmshape[1]
        ofmsize = ofmshape[0] * ofmshape[1]
        for fmind in range(nfm):
            ifm = inputs[fmind * ifmsize:(fmind + 1) * ifmsize]
            ofm = outputs[fmind * ofmsize:(fmind + 1) * ofmsize]
            maxfm = maxinds[fmind * ofmsize:(fmind + 1) * ofmsize]
            for dst in range(ofmsize):
                # For this output unit, get the corresponding receptive field
                # within the input feature map.
                rf = ifm.take(links[dst], axis=0)
                # Save the index of the maximum value.
                maxfm[dst] = rf.argmax(axis=0)
                # Set the pre-activations to the maximum value.
                maxvals = rf[maxinds[dst], range(rf.shape[1])]
                ofm[dst] = maxvals

    # Alternate implementation of max pooling bprop. To be deleted.
    def bprop_mpool2(self, inputs, outputs, error, berror, links, ifmshape,
                     ofmshape, fshape, padding, stride, nfm, maxinds):
        self.fill(berror, 0.0)
        ifmsize = ifmshape[0] * ifmshape[1]
        ofmsize = ofmshape[0] * ofmshape[1]
        for fmind in range(nfm):
            ifm = berror[fmind * ifmsize:(fmind + 1) * ifmsize]
            ofm = error[fmind * ofmsize:(fmind + 1) * ofmsize]
            maxfm = maxinds[fmind * ofmsize:(fmind + 1) * ofmsize]
            for dst in range(ofmsize):
                rflinks = links[dst]
                inds = rflinks.take(maxfm[dst], axis=0)
                ifm[inds, range(ifm.shape[1])] += ofm[dst]

    def fprop_apool(self, inputs, outputs, outputsbuf, links,
                    ifmshape, ofmshape, fshape, padding, stride, nfm):
        rinputs = self.hstack_maps(inputs, nfm)
        for dst in range(ofmshape[0] * ofmshape[1]):
            rf = rinputs.take(links[dst], axis=0)
            outputsbuf[dst] = rf.mean(axis=0)
        outputs[:] = self.vstack_maps(outputsbuf, nfm)

    def bprop_apool(self, outputs, error, berror, berrorbuf, links,
                    ifmshape, ofmshape, fshape, padding, stride, nfm):
        self.fill(berrorbuf, 0.0)
        error /= fshape[0] * fshape[1]
        rerror = self.hstack_maps(error, nfm)
        for dst in range(ofmshape[0] * ofmshape[1]):
            berrorbuf[links[dst]] += rerror[dst]
        berror[:] = self.vstack_maps(berrorbuf, nfm)

    def fprop_l2pool(self, inputs, outputs, outputsbuf, links,
                     ifmshape, ofmshape, fshape, padding, stride, nfm):
        rinputs = self.hstack_maps(inputs, nfm)
        for dst in range(ofmshape[0] * ofmshape[1]):
            rf = rinputs.take(links[dst], axis=0)
            outputsbuf[dst] = self.norm(rf, 2, axis=0)
        outputs[:] = self.vstack_maps(outputsbuf, nfm)

    def bprop_l2pool(self, inputs, outputs, error, berror, berrorbuf, links,
                     ifmshape, ofmshape, fshape, padding, stride,
                     nfm, prodbuf):
        rinputs = self.hstack_maps(inputs, nfm)
        routputs = self.hstack_maps(outputs, nfm)
        rerror = self.hstack_maps(error, nfm)
        self.fill(berrorbuf, 0.0)
        for dst in range(ofmshape[0] * ofmshape[1]):
            inds = links[dst]
            rf = rinputs.take(inds, axis=0)
            denom = routputs[dst].copy()
            # If the L2 norm is zero, the entire receptive field must be
            # zeros. In that case, we set the L2 norm to 1 before using
            # it to normalize the receptive field.
            denom[denom.raw() == 0] = 1
            self.divide(rf, denom, out=rf)
            self.multiply(
                rerror[dst:(dst + 1)].repeat(fshape[0] * fshape[1], axis=0),
                rf, out=prodbuf)
            berrorbuf[inds] += prodbuf
        berror[:] = self.vstack_maps(berrorbuf, nfm)

    def fprop_cmrnorm(self, inputs, outputs, ifmshape, nfm, ksize, alpha,
                      beta):
        (H, W, N) = (ifmshape[0], ifmshape[1], inputs.shape[1])
        rinputs = inputs._tensor.reshape((nfm, H, W, N))
        routputs = outputs._tensor.reshape((nfm, H, W, N))
        for i in range(nfm):
            x = rinputs[max(i-ksize/2, 0):min(i-ksize/2+ksize, nfm)]
            np.square(x).sum(axis=0, out=routputs[i])
        self.multiply(outputs, self.wrap(alpha), out=outputs)
        self.add(outputs, self.wrap(1.0), out=outputs)
        self.power(outputs, self.wrap(-beta), out=outputs)
        self.multiply(inputs, outputs, out=outputs)

    def bprop_cmrnorm_approx(self, inputs, outputs, error, berror, ifmshape,
                             nfm, ksize, alpha, beta, tempbuf):
        berror[:] = error

    def bprop_cmrnorm(self, inputs, outputs, error, berror, ifmshape, nfm,
                      ksize, alpha, beta, tempbuf):
        (H, W, N) = (ifmshape[0], ifmshape[1], inputs.shape[1])
        rinputs = inputs.reshape((nfm, H, W, N))
        rberror = berror.reshape((nfm, H, W, N))
        routputs = outputs.reshape((nfm, H, W, N))

        otemp = routputs.copy()
        # We can do this because rinputs[routputs == 0].sum() == 0
        otemp[otemp._tensor == 0] = 1.0
        self.divide(rinputs, otemp, out=otemp)
        itemp = rinputs.copy()
        # We can do this because routputs[rinputs == 0].sum() == 0
        itemp[itemp._tensor == 0] = 1.0
        self.divide(routputs, itemp, out=itemp)

        self.power(otemp, self.wrap(1.0 / beta), out=otemp)
        self.multiply(otemp, routputs, out=otemp)
        self.multiply(otemp, self.wrap(-2 * alpha * beta), out=otemp)
        self.fill(rberror, 0.0)

        for i in range(nfm):
            for j in range(max(i-ksize/2, 0), min(i-ksize/2+ksize, nfm)):
                self.multiply(otemp[i], rinputs[j], out=tempbuf)
                if i == j:
                    self.add(tempbuf, itemp[i], out=tempbuf)
                self.add(rberror[i], tempbuf, out=rberror[i])

        self.multiply(error, berror, out=berror)

    def fprop_fc(self, inputs, weights, out):
        self.dot(weights, inputs, out)

    def bprop_fc(self, deltas, weights, out):
        self.dot(weights.transpose(), deltas, out)

    def update_fc(self, deltas, inputs, out):
        self.dot(deltas, inputs.transpose(), out)

    def fprop_cmpool(self, inputs, weights, fmsize, out):
        for ofmind in range(weights.shape[1]):
            ofm = out[(ofmind * fmsize):((ofmind + 1) * fmsize)]
            self.fill(ofm, 0.0)
            for ifmind in range(weights.shape[0]):
                ifm = inputs[(ifmind * fmsize):((ifmind + 1) * fmsize)]
                ofm += ifm * weights[ifmind, ofmind]

    def bprop_cmpool(self, deltas, weights, fmsize, out):
        self.fprop_cmpool(deltas, weights.transpose(), fmsize, out)

    def update_cmpool(self, deltas, inputs, fmsize, updatebuf, out):
        self.fill(out, 0.0)
        for ofmind in range(out.shape[1]):
            ofmd = deltas[(ofmind * fmsize):((ofmind + 1) * fmsize)]
            for ifmind in range(out.shape[0]):
                ifm = inputs[(ifmind * fmsize):((ifmind + 1) * fmsize)]
                ofmd = ofmd.reshape((1, ofmd.shape[0] * ofmd.shape[1]))
                ifm = ifm.reshape((ifm.shape[0] * ifm.shape[1], 1))
                self.dot(ofmd, ifm, updatebuf)
                out[ifmind, ofmind] = updatebuf

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
            CPUTensor: The initialized weights
        """
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
            logger.info('generating %s uniform(%0.2f, %0.2f) weights.',
                        str(size), low, high)
            weights = self.uniform(low, high, size, dtype)
        elif (weight_params['type'] == 'gaussian' or
              weight_params['type'] == 'normal'):
            loc = 0.0
            scale = 1.0
            if 'loc' in weight_params:
                loc = weight_params['loc']
            if 'scale' in weight_params:
                scale = weight_params['scale']
            logger.info('generating %s normal(%0.2f, %0.2f) weights.',
                        str(size), loc, scale)
            weights = self.normal(loc, scale, size, dtype)
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
            elements = size[0] * size[1]
            nonzeros = size[0] * sparseness
            weights = np.zeros(size).flatten()
            nonzeroindex = np.random.permutation(elements)[0:nonzeros]
            weights[nonzeroindex] = 0.3 * np.random.randn(nonzeros)
            weights = weights.reshape(size)
            if size[0] == size[1]:
                temp = np.linalg.eig(weights)
                max_eig = np.max(np.absolute(temp[0]))
                logger.info('cpu: dividing by max eigenvalue %2.2f', max_eig)
                weights = self.tensor_cls(eigenvalue * weights / max_eig)
            else:
                logger.info('Matrix is non-square, no eigenvalue scaling.')
                weights = self.tensor_cls(weights)

        elif weight_params['type'] == 'node_normalized':
            # initialization is as discussed in Glorot2010
            scale = 1.0
            if 'scale' in weight_params:
                scale = weight_params['scale']
            logger.info('generating %s node_normalized(%0.2f) weights.',
                        str(size), scale)
            node_norm = scale * math.sqrt(6.0 / sum(size))
            weights = self.uniform(-node_norm, node_norm, size, dtype)
        else:
            raise AttributeError("invalid weight_params specified")
        if 'bias_init' in weight_params:
            # per append_bias() bias weights are in the last column
            logger.info('separately initializing bias weights to %0.2f',
                        weight_params['bias_init'])
            weights[:, -1] = weight_params['bias_init']
        return weights


# template for CPUDist (wrap MPI function calls so _tensor don't have to be
# exposed in layer code)
class CPUDist(CPU):

    def bcast(self, buf, rank=0):
        buf._tensor = MPI.COMM_WORLD.bcast(buf._tensor, rank)

# once CPUDist is implemented inherit from CPUDist


class CPUDataDist(CPU):

    """
    helper sub-class for data parallel implementations
    """

    def update_fc(self, deltas, inputs, out):
        super(CPUDataDist, self).update_fc(deltas, inputs, out)
        # trivial implementation below
        # could optimize by making each proc responsible for #params/comm.size
        # of the params
        out._tensor = MPI.COMM_WORLD.reduce(out.raw(), op=MPI.SUM, root=0)
        # This division by comm.size corresponds to following line in mlp bprop
        # self.backend.divide(error,
        #                    self.backend.wrap(targets.shape[
        #                                      targets.major_axis()]),
        #                    out=error)
        out._tensor = MPI.COMM_WORLD.bcast(out.raw())

    def update_conv(self, weights, inputs, error, updates, links, ifmshape,
                    ofmshape, ofmlocs, padding, stride, nifm, ngroups, fwidth,
                    updatebuf):
        super(CPUDataDist, self).update_conv(weights, inputs, error, updates,
                                             links, ifmshape, ofmshape,
                                             ofmlocs, padding, stride, nifm,
                                             ngroups, fwidth, updatebuf)
        updates._tensor = MPI.COMM_WORLD.reduce(updates.raw(), op=MPI.SUM,
                                                root=0)
        updates._tensor = MPI.COMM_WORLD.bcast(updates.raw())
