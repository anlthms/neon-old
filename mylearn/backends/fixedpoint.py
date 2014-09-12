"""
Custom fixed point backend and Tensor class.  Has configurable
integer and fraction bit width, rounding and overflow schemes.
"""

import logging
import numpy as np

from mylearn.backends.fixpt_cython import (fixed_from_float,
                                           fixed_from_float_array,
                                           fixed_to_float,
                                           fixed_to_float_array)
from mylearn.backends._numpy import Numpy, NumpyTensor

logger = logging.getLogger(__name__)

# underlying element type.  TODO: expose the cython declaration directly
elemtype_t = np.int64


class FixedPoint(Numpy):
    """
    Sets up a CPU based fixed point backend for matrix ops.
    """

    @staticmethod
    def zeros(shape, dtype=elemtype_t, sign_bit=True, int_bits=5, frac_bits=10,
              overflow=0, rounding=0):
        """
        Instantiates a new FixedPointTensor object whose elements are all set
        to zero.
            shape (int or sequence of ints): Shape of the new array.
            dtype (data-type, optional): Underlying element data-type.  For the
                                         moment this should not be used (we
                                         internally use integer storage, but we
                                         plan on allowing a custom fixpt dtype
                                         where we can incorporate the
                                         parameters below)
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
        # TODO: allow fixpt dtype and extract int_bits, frac_bits, and so forth
        # instead.
        return FixedPointTensor(np.zeros(shape, elemtype_t), sign_bit,
                                int_bits, frac_bits, overflow, rounding)

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
 
    @staticmethod
    def dot(a, b, out):
        naive_dot(a._tensor, b._tensor, out._tensor, out.sign_bit,
                  out.int_bits, out.frac_bits, out.overflow, out.rounding)


class FixedPointTensor(NumpyTensor):
    """
    CPU based configurable fixed point data structure.

    Arguments:
        obj (numpy.ndarray): the actual data values.  Python built-in
                             types like lists and tuples are also supported.
        dtype (data-type, optional): Underlying element data-type.  For the
                                     moment this should not be used (we
                                     internally use integer storage, but we
                                     plan on allowing a custom fixpt dtype
                                     where we can incorporate the
                                     parameters below)
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
    def __init__(self, obj, dtype=elemtype_t, sign_bit=True, int_bits=5,
                 frac_bits=10, overflow=0, rounding=0):
        super(FixedPointTensor, self).__init__(obj, dtype=np.float32)
        self._tensor = fixed_from_float_array(self._tensor, sign_bit,
                                              int_bits, frac_bits, overflow,
                                              rounding)
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
