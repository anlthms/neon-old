# coding: utf-8
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Custom flexible decimal point (Flexpoint™) backend and Tensor class.  Has
configurable integer and fraction bit width, rounding and overflow schemes.
"""

import logging
import numpy as np

from neon.backends.cpu import CPU, CPUTensor
from neon.backends.flexpt_cython import (flex_from_float,
                                         flex_from_float_array,
                                         flex_to_float,
                                         flex_to_float_array,
                                         naive_dot, elemtype, elemfloat,
                                         flexpt_dtype, fp_rescale_array)

logger = logging.getLogger(__name__)


class FlexpointTensor(CPUTensor):
    """
    CPU based configurable Flexpoint™ data structure.

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        dtype (flexpt_dtype): Specification of the parameters like integer and
                              fraction bits, overflow, and rounding handling.
        force_rescale (boolean, optional): if False (default) we may short
                                           circuit scaling of the object to
                                           our internal format if the input
                                           array is already of the expected
                                           type.  Set this to True to ignore
                                           the input type and force conversion
    """
    default_dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=11,
                                 overflow=0, rounding=0)

    def __init__(self, obj, dtype=None, force_rescale=False):
        if dtype is None:
            dtype = self.default_dtype
        if ((not force_rescale) and type(obj) == np.ndarray and (obj.dtype ==
                                                                 elemtype)):
            # already in the correct format, just assign to the _tensor
            self._tensor = obj
            self.shape = obj.shape
        elif (not force_rescale) and type(obj) == elemtype:
            # single element case
            self._tensor = np.array([[obj]], elemtype)
            self.shape = self._tensor.shape
        else:
            super(FlexpointTensor, self).__init__(obj, dtype=elemfloat)
            force_rescale = True
            if not self._tensor.flags['C_CONTIGUOUS']:
                self._tensor = np.ascontiguousarray(self._tensor)
        # ensure we can convert to a 2D representation
        # TODO: add support for 3D, 4D, etc.
        if self._tensor.ndim != 2:
            # check for single element or nx1 vector, and convert to 2D
            if self._tensor.ndim == 0 or self._tensor.ndim == 1:
                vec_len = 1
                if self._tensor.ndim == 1:
                    vec_len = self._tensor.shape[0]
                self._tensor = self._tensor.reshape([vec_len, 1])
                self.shape = (vec_len, 1)
            else:
                logger.error("Unsupported shape")
        if force_rescale:
            self._tensor = flex_from_float_array(self._tensor, dtype)
        self.dtype = dtype

    def __str__(self):
        return str(flex_to_float_array(np.ascontiguousarray(self._tensor),
                                       self.dtype))

    def __repr__(self):
        return ("%s(%s, dtype=%s)" %
                (self.__class__.__name__, str(self), str(self.dtype)))

    def __getitem__(self, key):
        return self.__class__(self._tensor[self._clean(key)], dtype=self.dtype)

    def __setitem__(self, key, value):
        clean_key = self._clean(key)
        self._tensor[clean_key] = self._clean(self.__class__(value))
        if isinstance(value, self.__class__):
            fp_rescale_array(self._tensor[clean_key], value.dtype, self.dtype)

    def asnumpyarray(self):
        return flex_to_float_array(np.ascontiguousarray(self._tensor),
                                   self.dtype)

    def transpose(self):
        return self.__class__(self._tensor.transpose(), dtype=self.dtype)

    def copy(self):
        return self.__class__(np.copy(self._tensor), dtype=self.dtype)


class Flexpoint(CPU):
    """
    Sets up a CPU based Flexpoint™ backend for matrix ops.

    We support the following attributes:

        sign_bit (bool, optional): If True (default) reserve one bit to
                                   storing sign information (pos/neg),
                                   otherwise only non-negative values will
                                   be supported.
        int_bits (int, optional): How many bits to reserve for storing the
                                  integer portion of the numeric value.
                                  If not specified, we use 5 bits.
        frac_bits (int, optional): How many bits to reserve for storing the
                                   fractional portion of the numeric value.
                                   If not specified, we use 10 bits.
        overflow (int, optional): How to handle values too large (positvely
                                  or negatively) to be stored in the given
                                  number of bits.  Possible choices are
                                  0: "saturate" (default), or 1: "wrap".
        rounding (int, optional): How to handle values that can't be stored
                                  exactly given the precision of our
                                  representation.  Possible choices are
                                  0: "truncate" (default), or 1: "nearest".
    """
    default_dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=11,
                                 overflow=0, rounding=0)
    tensor_cls = FlexpointTensor
    epsilon = 2**-10

    def empty(self, shape, dtype=None):
        """
        Instantiates a new FlexpointTensor object whose elements are all set
        to zero.

        Arguments:
            shape (int or sequence of ints): Shape of the new array.
            dtype (flexpt_dtype, optional): Specification of Flexpoint™
                                            parameters, passed through to the
                                            FlexpointTensor constructor.
                                            If None, we use the values
                                            specified in the default_dtype
                                            attribute.
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.empty(shape, dtype=elemfloat), dtype)

    def array(self, obj, dtype=None):
        """
        Instantiates a new FlexpointTensor object whose elements are set to the
        values of obj.

        Arguments:
            obj (array_like): input array object to construct from.  Can be
                              built-in python scalar or list (of lists), or a
                              numpy.ndarray
            dtype (flexpt_dtype, optional): Specification of Flexpoint™
                                            parameters, passed through to the
                                            FlexpointTensor constructor.
                                            If None, we use the values
                                            specified in the default_dtype
                                            attribute.
        """
        return self.tensor_cls(np.array(obj, dtype=elemfloat), dtype)

    def zeros(self, shape, dtype=None):
        """
        Instantiates a new FlexpointTensor object whose elements are all set
        to zero.

        Arguments:
            shape (int or sequence of ints): Shape of the new array.
            dtype (flexpt_dtype, optional): Specification of Flexpoint™
                                            parameters, passed through to the
                                            FlexpointTensor constructor.
                                            If None, we use the values
                                            specified in the default_dtype
                                            attribute.
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.zeros(shape, dtype=elemfloat), dtype)

    def ones(self, shape, dtype=None):
        """
        Instantiates a new FlexpointTensor object whose elements are all set
        to one.

        Arguments:
            shape (int or sequence of ints): Shape of the new array.
            dtype (flexpt_dtype, optional): Specification of Flexpoint™
                                            parameters, passed through to the
                                            FlexpointTensor constructor.
                                            If None, we use the values
                                            specified in the default_dtype
                                            attribute.
        """
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.ones(shape, dtype=elemfloat), dtype)

    def alloc(self, nrows, ncols, dtype=None):
        dtype = self.default_dtype_if_missing(dtype)
        return self.tensor_cls(np.zeros((nrows, ncols), dtype=elemfloat),
                               dtype)

    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (FlexpointTensor): left-hand side operand.
            right (FlexpointTensor): right-hand side operand.
            out (FlexpointTensor): where the result will be stored.

        Returns:
            FlexpointTensor: reference to out
        """
        np.equal(left._tensor, right._tensor, out._tensor)
        # rescale int64 0 or 1 value up to internal type
        fp_rescale_array(out._tensor, flexpt_dtype(True, 5, 0, 0, 0),
                         out.dtype)
        return out

    def not_equal(self, left, right, out):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (FlexpointTensor): left-hand side operand.
            right (FlexpointTensor): right-hand side operand.
            out (FlexpointTensor): where the result will be stored.

        Returns:
            FlexpointTensor: reference to out
        """
        np.not_equal(left._tensor, right._tensor, out._tensor)
        # rescale int64 0 or 1 value up to internal type
        fp_rescale_array(out._tensor, flexpt_dtype(True, 5, 0, 0, 0),
                         out.dtype)
        return out

    def greater(self, left, right, out):
        """
        Performs element-wise greater than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (FlexpointTensor): left-hand side operand.
            right (FlexpointTensor): right-hand side operand.
            out (FlexpointTensor): where the result will be stored.

        Returns:
            FlexpointTensor: reference to out
        """
        np.greater(left._tensor, right._tensor, out._tensor)
        # rescale int64 0 or 1 value up to internal type
        fp_rescale_array(out._tensor, flexpt_dtype(True, 5, 0, 0, 0),
                         out.dtype)
        return out

    def greater_equal(self, left, right, out):
        """
        Performs element-wise greater than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (FlexpointTensor): left-hand side operand.
            right (FlexpointTensor): right-hand side operand.
            out (FlexpointTensor): where the result will be stored.

        Returns:
            FlexpointTensor: reference to out
        """
        np.greater_equal(left._tensor, right._tensor, out._tensor)
        # rescale int64 0 or 1 value up to internal type
        fp_rescale_array(out._tensor, flexpt_dtype(True, 5, 0, 0, 0),
                         out.dtype)
        return out

    def less(self, left, right, out):
        """
        Performs element-wise less than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (FlexpointTensor): left-hand side operand.
            right (FlexpointTensor): right-hand side operand.
            out (FlexpointTensor): where the result will be stored.

        Returns:
            FlexpointTensor: reference to out
        """
        np.less(left._tensor, right._tensor, out._tensor)
        # rescale int64 0 or 1 value up to internal type
        fp_rescale_array(out._tensor, flexpt_dtype(True, 5, 0, 0, 0),
                         out.dtype)
        return out

    def less_equal(self, left, right, out):
        """
        Performs element-wise less than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (FlexpointTensor): left-hand side operand.
            right (FlexpointTensor): right-hand side operand.
            out (FlexpointTensor): where the result will be stored.

        Returns:
            FlexpointTensor: reference to out
        """
        np.less_equal(left._tensor, right._tensor, out._tensor)
        # rescale int64 0 or 1 value up to internal type
        fp_rescale_array(out._tensor, flexpt_dtype(True, 5, 0, 0, 0),
                         out.dtype)
        return out

    def norm(self, tsr, order=None, axis=None, out=None):
        """
        Calculates and returns the vector p-norms of the FlexpointTensor along
        the specified axis.  The p-norm is defined on vector A as
        :math:`||A||_p = \sum_i(|A_i|^p)^{1/p}`.

        Arguments:
            tsr (FlexpointTensor): the FlexpointTensor on which to find the
                                   norms
            order (int): The order or p upon which the norm is calculated.
                         Valid values include:
                         None, inf, -inf, 0, 1, -1, 2, -2, ...
            axis (int): The axis along which to compute vector norms.
            out (FlexpointTensor, optional): where to write the results to.
                                             Must be of the expected result
                                             shape.  If not specified, a new
                                             buffer is created and returned.

        Returns:
            FlexpointTensor: p-norm of tsr along the specified axis.

        See Also:
            `numpy.linalg.norm`
        """
        if not isinstance(axis, int):
            raise AttributeError("invalid axis value: %s", axis)
        if order == float('Inf'):
            res = np.max(np.abs(flex_to_float_array(tsr._tensor, tsr.dtype)),
                         axis)
        elif order == float('-Inf'):
            res = np.min(np.abs(flex_to_float_array(tsr._tensor, tsr.dtype)),
                         axis)
        elif order == 0:
            res = np.sum(tsr._tensor != 0, axis)
        else:
            res = np.sum(np.abs(flex_to_float_array(tsr._tensor, tsr.dtype)) **
                         order, axis) ** (1.0 / order)
        if out is None:
            out = self.array(res)
        else:
            res = self.array(res)
            out._tensor = res
            out.shape = res.shape
            fp_rescale_array(out._tensor, res.dtype, out.dtype)
        return out

    def wrap(self, obj, dtype=None):
        if dtype is None:
            dtype = self.suggest_dtype(obj)
        return self.tensor_cls(obj, dtype)

    def suggest_dtype(self, obj, max_word_bits=16):
        """
        Attempts to infer a reasonable dtype that balances handling the entire
        data input range without losing too much precision.
        """
        sign_bit = False
        int_bits = 0
        frac_bits = max_word_bits
        overflow = 0
        rounding = 0
        min_val = obj
        max_val = obj
        if isinstance(obj, (list, tuple, np.ndarray)):
            min_val = np.min(obj)
            max_val = np.max(obj)
        if min_val < 0:
            sign_bit = True
            frac_bits -= 1
            max_val = abs(max_val)
        while ((sign_bit + int_bits) < max_word_bits and
                max_val >= (1 << int_bits)):
            int_bits += 1
            frac_bits -= 1
        return flexpt_dtype(sign_bit, int_bits, frac_bits, overflow, rounding)

    def display(self, value, sign_bit=True, int_bits=5, frac_bits=10,
                overflow=0, rounding=0):
        """
        Helper to print a representation of the given value in various forms
        according to the parameters specified.

        Arguments:
            value (float): the actual data value.
            sign_bit (bool, optional): If True (default) reserve one bit to
                                       storing sign information (pos/neg),
                                       otherwise only non-negative values will
                                       be supported.
            int_bits (int, optional): How many bits to reserve for storing the
                                      integer portion of the numeric value.
                                      If not specified, we use 5 bits.
            frac_bits (int, optional): How many bits to reserve for storing the
                                       fractional portion of the numeric value.
                                       If not specified, we use 10 bits.
            overflow (int, optional): How to handle values too large (positvely
                                      or negatively) to be stored in the given
                                      number of bits.  Possible choices are
                                      0: "saturate" (default), or 1: "wrap".
            rounding (int, optional): How to handle values that can't be stored
                                      exactly given the precision of our
                                      representation.  Possible choices are
                                      0: "truncate" (default), 1: "nearest".
        Return:
            str: pretty-print formatted representation
        """
        from bitstring import BitArray
        stored_val = flex_from_float(value, sign_bit, int_bits, frac_bits,
                                     overflow, rounding)
        conv_val = flex_to_float(stored_val, sign_bit, int_bits, frac_bits,
                                 overflow, rounding)
        sign_val = ''
        if sign_bit:
            if conv_val > 0:
                sign_val = '+'
            elif conv_val < 0:
                sign_val = '-'
        return ("raw value:\t%f (decimal)\t%s (binary)\n"
                "stored int:\t%d (decimal)\t%s (binary)\n"
                "flexpt value:\t%s%f (Q%d.%d %ssigned decimal)" %
                (value, BitArray(float=value, length=32).bin, stored_val,
                 BitArray(float=stored_val, length=32).bin, sign_val,
                 conv_val, int_bits, frac_bits, "" if sign_bit else "un"))

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
            dtype (flexpt_dtype, optional): Specification of flexpoint
                                            parameters, passed through to the
                                            FlexpointTensor constructor.
                                            If None, we use the values
                                            specified in the default_dtype
                                            attribute.

        Returns:
            FlexPointTensor: Of specified size filled with these random
                             numbers.
        """
        return FlexpointTensor(np.random.uniform(low, high, size), dtype)

    def append_bias(self, x):
        """
        Adds a bias column of ones to FlexpointTensor x,
        returning a new FlexpointTensor.
        """
        float_x = flex_to_float_array(x._tensor, x.dtype)
        bias = np.ones((x.shape[0], 1), dtype=elemfloat)
        return FlexpointTensor(np.concatenate((float_x, bias), axis=1),
                               x.dtype)

    def dot(self, a, b, out):
        if not a._tensor.flags['C_CONTIGUOUS']:
                a._tensor = np.ascontiguousarray(a._tensor)
        if not b._tensor.flags['F_CONTIGUOUS']:
                b._tensor = np.asfortranarray(b._tensor)
        if not out._tensor.flags['C_CONTIGUOUS']:
                out._tensor = np.ascontiguousarray(out._tensor)
        naive_dot(a._tensor, b._tensor, out._tensor, a.dtype, b.dtype,
                  out.dtype)

    def scale_to_largest(self, a, b):
        """
        Helper that ensures operands are on the same scale by (potentially)
        copying and upcasting one of the operands.

        Arguments:
            a (FlexpointTensor): left operand
            b (FlexpointTensor): right operand

        Returns:
            list: 3-tuple containing common dtype, then a and b tensors
        """
        out_a = a._tensor
        out_b = b._tensor
        out_dtype = a.dtype
        if a.dtype['int_bits'] > b.dtype['int_bits']:
            # scale b to a
            out_b = np.copy(b._tensor)
            fp_rescale_array(out_b, b.dtype, a.dtype)
        elif a.dtype['int_bits'] < b.dtype['int_bits']:
            # scale a to b
            out_a = np.copy(a._tensor)
            fp_rescale_array(out_a, a.dtype, b.dtype)
            out_dtype = b.dtype
        elif a.dtype['frac_bits'] > b.dtype['frac_bits']:
            # same int_bits, but scale b to a to increase precision
            out_b = np.copy(b._tensor)
            fp_rescale_array(out_b, b.dtype, a.dtype)
        elif a.dtype['frac_bits'] < b.dtype['frac_bits']:
            # same int_bits, but scale a to b to increase precision
            out_a = np.copy(a._tensor)
            fp_rescale_array(out_a, a.dtype, b.dtype)
            out_dtype = b.dtype
        return (out_dtype, out_a, out_b)

    def add(self, a, b, out):
        if not isinstance(a, self.tensor_cls):
            a = self.wrap(a)
        if not isinstance(b, self.tensor_cls):
            b = self.wrap(b)
        in_dtype, a_tensor, b_tensor = self.scale_to_largest(a, b)
        np.add(a_tensor, b_tensor, out._tensor)
        fp_rescale_array(out._tensor, in_dtype, out.dtype)

    def subtract(self, a, b, out):
        if not isinstance(a, self.tensor_cls):
            a = self.wrap(a)
        if not isinstance(b, self.tensor_cls):
            b = self.wrap(b)
        in_dtype, a_tensor, b_tensor = self.scale_to_largest(a, b)
        np.subtract(a_tensor, b_tensor, out._tensor)
        fp_rescale_array(out._tensor, in_dtype, out.dtype)

    def multiply(self, a, b, out):
        if not isinstance(a, self.tensor_cls):
            a = self.wrap(a)
        if not isinstance(b, self.tensor_cls):
            b = self.wrap(b)
        # for multiplication we don't need matching scales to start
        np.multiply(a._tensor, b._tensor, out._tensor)
        tmp_dtype = flexpt_dtype(a.dtype['sign_bit'],
                                 a.dtype['int_bits'] + b.dtype['int_bits'],
                                 a.dtype['frac_bits'] + b.dtype['frac_bits'],
                                 a.dtype['overflow'], a.dtype['rounding'])
        fp_rescale_array(out._tensor, tmp_dtype, out.dtype)

    def divide(self, a, b, out):
        if not isinstance(a, self.tensor_cls):
            a = self.wrap(a)
        if not isinstance(b, self.tensor_cls):
            b = self.wrap(b)
        # for division, shift the numerator to required scale first
        # then do integer division.
        # required scale assuming out has f frac bits, a has m frac bits, and
        # b has n frac bits is f - (m - n)
        a_tensor = np.copy(a._tensor)
        tmp_dt = a.dtype.copy()
        tmp_dt['frac_bits'] += out.dtype['frac_bits'] - (a.dtype['frac_bits'] -
                                                         b.dtype['frac_bits'])
        fp_rescale_array(a_tensor, a.dtype, tmp_dt)
        np.divide(a_tensor, b._tensor, out._tensor)

    def reciprocal(self, a, out):
        self.divide(self.wrap(1.0, out.dtype), a, out)

    def log(self, x, out):
        # for the moment we punt on a Flexpoint™ log, just do an
        # expensive conversion to/from floating point.
        # See: http://lib.tkk.fi/Diss/2005/isbn9512275279/article8.pdf
        tmp = flex_to_float_array(x._tensor, x.dtype)
        np.log(tmp, tmp)
        out._tensor = flex_from_float_array(tmp, out.dtype)

    def exp(self, x, out):
        # for the moment we punt on a Flexpoint™ exponent, just do an
        # expensive conversion to/from floating point.
        # See: http://lib.tkk.fi/Diss/2005/isbn9512275279/article8.pdf
        tmp = flex_to_float_array(x._tensor, x.dtype)
        np.exp(tmp, tmp)
        out._tensor = flex_from_float_array(tmp, out.dtype)

    def logistic(self, x, out):
        # for the moment we punt on a Flexpoint™ logistic, just do an
        # expensive conversion to/from floating point.
        # See: http://lib.tkk.fi/Diss/2005/isbn9512275279/article8.pdf
        tmp = flex_to_float_array(x._tensor, x.dtype)
        tmp = 1.0 / (1.0 + np.exp(- tmp))
        out._tensor = flex_from_float_array(tmp, out.dtype)

    def clip(self, a, a_min, a_max, out=None):
        if out is None:
            out = FlexpointTensor(np.empty_like(a._tensor), a.dtype)
        np.clip(a._tensor, flex_from_float(a_min, a.dtype),
                flex_from_float(a_max, a.dtype), out._tensor)
        fp_rescale_array(out._tensor, a.dtype, out.dtype)
        return out

    def sum(self, x, axes=None, dtype='float32', out=None, keepdims=True):
        if x is None:
            return float('NaN')
        res = np.sum(flex_to_float_array(x._tensor, x.dtype), axes, dtype,
                     out._tensor, keepdims)
        if axes is None and not keepdims:
            return res
        else:
            return FlexpointTensor(res, x.dtype)

    def mean(self, x, axes=None, dtype='float32', out=None, keepdims=True):
        if x is None:
            return float('NaN')
        res = np.mean(flex_to_float_array(x._tensor, x.dtype), axes, dtype,
                      out._tensor, keepdims)
        if axes is None and not keepdims:
            return res
        else:
            return FlexpointTensor(res, x.dtype)

    def min(self, x, axes=None, out=None, keepdims=True):
        if x is None:
            return float('NaN')
        res = np.min(flex_to_float_array(x._tensor, x.dtype), axes,
                     out._tensor, keepdims)
        if axes is None and not keepdims:
            return res
        else:
            return FlexpointTensor(res, x.dtype)

    def max(self, x, axes=None, out=None, keepdims=True):
        if x is None:
            return float('NaN')
        res = np.max(flex_to_float_array(x._tensor, x.dtype), axes,
                     out._tensor, keepdims)
        if axes is None and not keepdims:
            return res
        else:
            return FlexpointTensor(res, x.dtype)

    def argmin(self, tsr, axis, out):
        """
        Calculates the indices of the minimal element value along the specified
        axis.  If multiple elements contain the minimum, only the elements of
        the first are returned.

        Arguments:
            tsr (FlexpointTensor): The FlexpointTensor on which to find the
                                   minimum indices
            axis (int): The dimension along which to find the minimum.  If set
                        to None, find the overall minimum index of a flattened
                        representation of tsr.
            out (FlexpointTensor): Where to store the result.  Should be of the
                                   appropriate type and expected shape

        Returns:
            FlexpointTensor: reference to out
        """
        out._tensor = flex_from_float_array(np.argmin(tsr._tensor, axis),
                                            out.dtype)
        out.shape = out._tensor.shape
        return out

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

        Returns:
            FlexpointTensor: reference to out
        """
        out._tensor = flex_from_float_array(np.argmax(tsr._tensor, axis),
                                            out.dtype)
        out.shape = out._tensor.shape
        return out

    def fabs(self, x, out=None):
        if out is not None:
            res = np.fabs(x._tensor, out._tensor)
        else:
            # np.fabs changes dtype to float64 by default, so cast this back
            # to our internal representation
            res = elemtype(np.fabs(x._tensor))
        return FlexpointTensor(res, x.dtype)
