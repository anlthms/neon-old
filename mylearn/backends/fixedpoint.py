"""
Custom fixed point backend and Tensor class.  Has configurable
integer and fraction bit width, rounding and overflow schemes.
"""

import logging
import numpy as np

from mylearn.backends.fixpt_cython import (fixed_from_float,
                                           fixed_from_float_array,
                                           fixed_to_float,
                                           fixed_to_float_array,
                                           naive_dot)
from mylearn.backends._numpy import Numpy, NumpyTensor

logger = logging.getLogger(__name__)

# underlying element type.  TODO: expose the cython declaration directly
elemtype_t = np.int64


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
    sign_bit = True
    int_bits = 5
    frac_bits = 10
    overflow = 0
    rounding = 0

    @classmethod
    def zeros(cls, shape):
        """
        Instantiates a new FixedPointTensor object whose elements are all set
        to zero.

        Arguments:
            shape (int or sequence of ints): Shape of the new array.
        """
        return FixedPointTensor(np.zeros(shape, dtype=np.float32),
                                sign_bit=cls.sign_bit, int_bits=cls.int_bits,
                                frac_bits=cls.frac_bits,
                                overflow=cls.overflow, rounding=cls.rounding)

    @classmethod
    def array(cls, obj):
        return FixedPointTensor(np.array(obj, dtype=np.float32),
                                sign_bit=cls.sign_bit, int_bits=cls.int_bits,
                                frac_bits=cls.frac_bits,
                                overflow=cls.overflow, rounding=cls.rounding)

    @staticmethod
    def wrap(obj):
        return FixedPointTensor(obj)

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

    def uniform(self, low=0.0, high=1.0, size=1):
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
            FixedPointTensor: Of specified size filled with these random
                         numbers.
        """
        return FixedPointTensor(np.random.uniform(low, high, size))

    def normal(self, loc=0.0, scale=1.0, size=1):
        """
        Gaussian/Normal random number sample generation

        Arguments:
            loc (float, optional): Where to center distribution.  Defaults
                                   to 0.0
            scale (float, optional): Standard deviaion.  Defaults to 1.0
            size (array_like or int, optional): Shape of generated samples

        Returns:
            FixedPointTensor: Of specified size filled with these random
                         numbers.
        """
        return FixedPointTensor(np.random.normal(loc, scale, size))

    @staticmethod
    def append_bias(x):
        """
        Adds a bias column of ones to FixedPointTensor x,
        returning a new FixedPointTensor.
        """
        float_x = fixed_to_float_array(x._tensor, x.sign_bit, x.int_bits,
                                       x.frac_bits, x.overflow, x.rounding)
        bias = np.ones((x.shape[0], 1), dtype=np.float32)
        return FixedPointTensor(np.concatenate((float_x, bias), axis=1))

    @staticmethod
    def dot(a, b, out):
        if a._tensor.flags['F_CONTIGUOUS']:
                a._tensor = np.ascontiguousarray(a._tensor)
        if b._tensor.flags['C_CONTIGUOUS']:
                b._tensor = np.asfortranarray(b._tensor)
        if out._tensor.flags['F_CONTIGUOUS']:
                out._tensor = np.ascontiguousarray(out._tensor)
        naive_dot(a._tensor, b._tensor, out._tensor, out.sign_bit,
                  out.int_bits, out.frac_bits, out.overflow, out.rounding)


class FixedPointTensor(NumpyTensor):
    """
    CPU based configurable fixed point data structure.

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        sign_bit (bool, optional): Set to True (default) for signed fixed point
                                   numbers, False for unsigned.
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
    """
    # TODO: convert passing all fixed point parameters individually into
    # a customized dtype?
    def __init__(self, obj, dtype=np.float32, sign_bit=True, int_bits=5,
                 frac_bits=10, overflow=0, rounding=0):
        if type(obj) == np.ndarray and obj.dtype == elemtype_t:
            self._tensor = obj
            self.shape = obj.shape
        else:
            super(FixedPointTensor, self).__init__(obj, dtype=dtype)
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
            self._tensor = fixed_from_float_array(self._tensor, sign_bit,
                                                  int_bits, frac_bits,
                                                  overflow, rounding)
        self.sign_bit = sign_bit
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.overflow = overflow
        self.rounding = rounding

    def __str__(self):
        return str(fixed_to_float_array(self._tensor, self.sign_bit,
                                        self.int_bits, self.frac_bits,
                                        self.overflow, self.rounding))

    def __repr__(self):
        return ("%s(%s, sign_bit=%s, int_bits=%d, frac_bits=%d, overflow=%d,"
                " rounding=%d)" %
                (self.__class__.__name__, str(self), self.sign_bit,
                 self.int_bits, self.frac_bits, self.overflow, self.rounding))
