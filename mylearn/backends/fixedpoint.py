"""
Custom fixed point backend and Tensor class.  Has configurable
integer and fraction bit width, rounding and overflow schemes.
"""

import logging
import math
import numpy as np

from mylearn.backends.fixpt_cython import (fixed_from_float,
                                           fixed_from_float_array,
                                           fixed_to_float,
                                           fixed_to_float_array,
                                           naive_dot, elemtype, elemfloat,
                                           fixpt_dtype, fp_rescale_array,
                                           fp_rescale)
from mylearn.backends._numpy import Numpy, NumpyTensor

logger = logging.getLogger(__name__)


class FixedPointTensor(NumpyTensor):
    """
    CPU based configurable fixed point data structure.

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        dtype (fixpt_dtype): Specification of the parameters like integer and
                             fraction bits, overflow, and rounding handling.
    """
    def __init__(self, obj, dtype):
        dtype = FixedPoint.default_dtype_if_missing(dtype)
        if type(obj) == np.ndarray and obj.dtype == elemtype:
            self._tensor = obj
            self.shape = obj.shape
        else:
            super(FixedPointTensor, self).__init__(obj, dtype=elemfloat)
            if self._tensor.ndim != 2:
                # TODO: add support for vectors, 3D, 4D, etc.
                # for now we just special case a 1x1 ndarray
                if self._tensor.ndim == 0 or (self._tensor.ndim == 1 and
                                              self._tensor.shape[0] == 1):
                    self._tensor = self._tensor.reshape([1, 1])
                    self.shape = [1, 1]
                else:
                    logger.error("Unsupported shape")
            if not self._tensor.flags['C_CONTIGUOUS']:
                self._tensor = np.ascontiguousarray(self._tensor)
            self._tensor = fixed_from_float_array(self._tensor, dtype)
        self.dtype = dtype

    def __str__(self):
        return str(fixed_to_float_array(np.ascontiguousarray(self._tensor),
                                        self.dtype))

    def __repr__(self):
        return ("%s(%s, dtype=%s)" %
                (self.__class__.__name__, str(self), str(self.dtype)))

    def __getitem__(self, key):
        return self.__class__(self._tensor[self._clean(key)], dtype=self.dtype)

    def T(self):  # flake8: noqa
        return self.__class__(self._tensor.T, dtype=self.dtype)

    def copy(self):
        return self.__class__(np.copy(self._tensor), dtype=self.dtype)


class FixedPoint(Numpy):
    """
    Sets up a CPU based fixed point backend for matrix ops.

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
    default_dtype = fixpt_dtype(sign_bit=True, int_bits=4, frac_bits=11,
                            overflow=0, rounding=0)
    tensor_cls = FixedPointTensor
    epsilon = 2**-10

    @classmethod
    def zeros(cls, shape, dtype=None):
        """
        Instantiates a new FixedPointTensor object whose elements are all set
        to zero.

        Arguments:
            shape (int or sequence of ints): Shape of the new array.
            dtype (fixpt_dtype, optional): Specification of fixedpoint
                                           parameters, passed through to the
                                           FixedPointTensor constructor.
                                           If None, we use the values
                                           specified in the default_dtype
                                           attribute.
        """
        dtype = cls.default_dtype_if_missing(dtype)
        return FixedPointTensor(np.zeros(shape, dtype=elemfloat), dtype)

    @classmethod
    def array(cls, obj, dtype=None):
        return FixedPointTensor(np.array(obj, dtype=elemfloat), dtype)

    @classmethod
    def wrap(cls, obj, dtype=None):
        if dtype is None:
            dtype = cls.suggest_dtype(obj)
        return FixedPointTensor(obj, dtype)

    @classmethod
    def suggest_dtype(cls, obj, max_word_bits=16):
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
            min_val = min(obj)
            max_val = max(obj)
        if min_val < 0:
            sign_bit = True
            frac_bits -= 1
            max_val = abs(max_val)
        while ((sign_bit + int_bits) < max_word_bits and
                max_val >= (1 << int_bits)):
            int_bits += 1
            frac_bits -= 1
        return fixpt_dtype(sign_bit, int_bits, frac_bits, overflow, rounding)

    @staticmethod
    def display(value, sign_bit=True, int_bits=5, frac_bits=10,
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
        stored_val = fixed_from_float(value, sign_bit, int_bits, frac_bits,
                                      overflow, rounding)
        conv_val = fixed_to_float(stored_val, sign_bit, int_bits, frac_bits,
                                  overflow, rounding)
        sign_val = ''
        if sign_bit:
            if conv_val > 0:
                sign_val = '+'
            elif conv_val < 0:
                sign_val = '-'
        return ("raw value:\t%f (decimal)\t%s (binary)\n"
                "stored int:\t%d (decimal)\t%s (binary)\n"
                "fixpt value:\t%s%f (Q%d.%d %ssigned decimal)" %
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
            dtype (fixpt_dtype, optional): Specification of fixedpoint
                                           parameters, passed through to the
                                           FixedPointTensor constructor.
                                           If None, we use the values
                                           specified in the default_dtype
                                           attribute.

        Returns:
            FixedPointTensor: Of specified size filled with these random
                         numbers.
        """
        return FixedPointTensor(np.random.uniform(low, high, size), dtype)

    @staticmethod
    def append_bias(x):
        """
        Adds a bias column of ones to FixedPointTensor x,
        returning a new FixedPointTensor.
        """
        float_x = fixed_to_float_array(x._tensor, x.dtype)
        bias = np.ones((x.shape[0], 1), dtype=elemfloat)
        return FixedPointTensor(np.concatenate((float_x, bias), axis=1),
                                x.dtype)

    @staticmethod
    def dot(a, b, out):
        if not a._tensor.flags['C_CONTIGUOUS']:
                a._tensor = np.ascontiguousarray(a._tensor)
        if not b._tensor.flags['F_CONTIGUOUS']:
                b._tensor = np.asfortranarray(b._tensor)
        if not out._tensor.flags['C_CONTIGUOUS']:
                out._tensor = np.ascontiguousarray(out._tensor)
        naive_dot(a._tensor, b._tensor, out._tensor, a.dtype, b.dtype,
                  out.dtype)

    @classmethod
    def scale_to_largest(cls, a, b):
        """
        Helper that ensures operands are on the same scale by (potentially)
        copying and upcasting one of the operands.

        Arguments:
            a (FixedPointTensor): left operand
            b (FixedPointTensor): right operand

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

    @classmethod
    def add(cls, a, b, out):
        in_dtype, a_tensor, b_tensor = cls.scale_to_largest(a, b)
        np.add(a_tensor, b_tensor, out._tensor)
        fp_rescale_array(out._tensor, in_dtype, out.dtype)

    @classmethod
    def subtract(cls, a, b, out):
        in_dtype, a_tensor, b_tensor = cls.scale_to_largest(a, b)
        np.subtract(a_tensor, b_tensor, out._tensor)
        fp_rescale_array(out._tensor, in_dtype, out.dtype)

    @classmethod
    def multiply(cls, a, b, out):
        # for multiplication we don't need matching scales to start
        np.multiply(a._tensor, b._tensor, out._tensor)
        tmp_dtype = fixpt_dtype(a.dtype['sign_bit'],
                                a.dtype['int_bits'] + b.dtype['int_bits'],
                                a.dtype['frac_bits'] + b.dtype['frac_bits'],
                                a.dtype['overflow'], a.dtype['rounding'])
        fp_rescale_array(out._tensor, tmp_dtype, out.dtype)

    @classmethod
    def divide(cls, a, b, out):
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

    @classmethod
    def reciprocal(cls, a, out):
        cls.divide(cls.wrap(1.0, out.dtype), a, out)

    @classmethod
    def greater(cls, a, b, out):
        np.greater(a._tensor, b._tensor, out._tensor)
        # greater result stored as int64 with 0 or 1 values.  Just need to
        # shift accordingly
        tmp_dtype = fixpt_dtype(True, 5, 0, 0, 0)
        fp_rescale_array(out._tensor, tmp_dtype, out.dtype)

    @classmethod
    def log(cls, x, out):
        # for the moment we punt on a fixed point exponent, just do an
        # expensive conversion to/from floating point.
        # See: http://lib.tkk.fi/Diss/2005/isbn9512275279/article8.pdf
        tmp = fixed_to_float_array(x._tensor, x.dtype)
        np.log(tmp, tmp)
        out._tensor = fixed_from_float_array(tmp, out.dtype)

    @classmethod
    def exp(cls, x, out):
        # for the moment we punt on a fixed point exponent, just do an
        # expensive conversion to/from floating point.
        # See: http://lib.tkk.fi/Diss/2005/isbn9512275279/article8.pdf
        tmp = fixed_to_float_array(x._tensor, x.dtype)
        np.exp(tmp, tmp)
        out._tensor = fixed_from_float_array(tmp, out.dtype)

    @classmethod
    def logistic(cls, x, out):
        # for the moment we punt on a fixed point exponent, just do an
        # expensive conversion to/from floating point.
        # See: http://lib.tkk.fi/Diss/2005/isbn9512275279/article8.pdf
        tmp = fixed_to_float_array(x._tensor, x.dtype)
        tmp = Numpy.wrap(tmp)
        Numpy.logistic(tmp, tmp)
        out._tensor = fixed_from_float_array(tmp._tensor, out.dtype)

    @staticmethod
    def clip(a, a_min, a_max, out=None):
        if out is None:
            out = FixedPointTensor(np.empty_like(a._tensor), a.dtype)
        np.clip(a._tensor, fixed_from_float(a_min, a.dtype),
                fixed_from_float(a_max, a.dtype), out._tensor)
        fp_rescale_array(out._tensor, a.dtype, out.dtype)
        return out

    @staticmethod
    def mean(x, axis=None, dtype=np.float32, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.mean(fixed_to_float_array(x._tensor, x.dtype), axis, dtype,
                      out, keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return FixedPointTensor(res, x.dtype)

    @staticmethod
    def min(x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.min(fixed_to_float_array(x._tensor, x.dtype), axis, out,
                     keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return FixedPointTensor(res, x.dtype)

    @staticmethod
    def max(x, axis=None, out=None, keepdims=False):
        if x is None:
            return float('NaN')
        res = np.max(fixed_to_float_array(x._tensor, x.dtype), axis, out,
                     keepdims)
        if axis is None and not keepdims:
            return res
        else:
            return FixedPointTensor(res, x.dtype)

    @staticmethod
    def fabs(x, out=None):
        if out is not None:
            res = np.fabs(x._tensor, out._tensor)
        else:
            # np.fabs changes dtype to float64 by default, so cast this back
            # to our internal representation
            res = elemtype(np.fabs(x._tensor))
        return FixedPointTensor(res, x.dtype)
