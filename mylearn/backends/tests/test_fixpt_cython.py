#!/usr/bin/env/python

import numpy as np

from mylearn.backends.fixpt_cython import (fixed_from_float, fixed_to_float,
                                           fp_rescale, naive_dot)
from mylearn.util.testing import assert_tensor_equal

# defaults
DEFAULT_FP_VALS = {
    "sign_bit": True,
    "int_bits": 5,
    "frac_bits": 10,
    "overflow": 0,  # 0 == OFL_SATURATE
    "rounding": 0  # 0 == RND_TRUNCATE
}


def create(val, sign_bit=DEFAULT_FP_VALS['sign_bit'],
           int_bits=DEFAULT_FP_VALS['int_bits'],
           frac_bits=DEFAULT_FP_VALS['frac_bits'],
           overflow=DEFAULT_FP_VALS['overflow'],
           rounding=DEFAULT_FP_VALS['rounding']):
    return fixed_from_float(val, sign_bit, int_bits, frac_bits, overflow,
                            rounding)


def as_decimal(val, sign_bit=DEFAULT_FP_VALS['sign_bit'],
               int_bits=DEFAULT_FP_VALS['int_bits'],
               frac_bits=DEFAULT_FP_VALS['frac_bits'],
               overflow=DEFAULT_FP_VALS['overflow'],
               rounding=DEFAULT_FP_VALS['rounding']):
    if type(val) == np.ndarray:
        return as_decimal_array(val, sign_bit, int_bits, frac_bits, overflow,
                                rounding)
    else:
        return as_decimal_scalar(val, sign_bit, int_bits, frac_bits, overflow,
                                 rounding)


def as_decimal_scalar(val, sign_bit=DEFAULT_FP_VALS['sign_bit'],
                      int_bits=DEFAULT_FP_VALS['int_bits'],
                      frac_bits=DEFAULT_FP_VALS['frac_bits'],
                      overflow=DEFAULT_FP_VALS['overflow'],
                      rounding=DEFAULT_FP_VALS['rounding']):
        return fixed_to_float(val, sign_bit, int_bits, frac_bits, overflow,
                              rounding)


as_decimal_array = np.vectorize(as_decimal_scalar, otypes='f')


def scale(val, in_sign_bit=DEFAULT_FP_VALS['sign_bit'],
          in_int_bits=DEFAULT_FP_VALS['int_bits'],
          in_frac_bits=DEFAULT_FP_VALS['frac_bits'],
          overflow=DEFAULT_FP_VALS['overflow'],
          rounding=DEFAULT_FP_VALS['rounding'],
          out_sign_bit=DEFAULT_FP_VALS['sign_bit'],
          out_int_bits=DEFAULT_FP_VALS['int_bits'],
          out_frac_bits=DEFAULT_FP_VALS['frac_bits']):
    return fp_rescale(val, in_sign_bit, in_int_bits, in_frac_bits, overflow,
                      rounding, out_sign_bit, out_int_bits, out_frac_bits)


def test_creation_and_rep():
    params = {
        "sign_bit": False,
        "int_bits": 3,
        "frac_bits": 9,
        "overflow": 0,
        "rounding": 0,
    }
    assert as_decimal(create(1.0, **params), **params) == 1.0


def test_defaulting():
    params = {
        "int_bits": 4,
        "sign_bit": 0,
        "overflow": 0,
        "frac_bits": 3,
    }
    assert as_decimal(create(9.0, **params), **params) == 9.0


def test_fracbit_rep():
    params = {
        "sign_bit": True,
        "int_bits": 1,
        "rounding": 1,  # RND_NEAREST
    }
    for frac_bits in range(0, 6):
        params["frac_bits"] = frac_bits
        x = create(1.1, **params)
        exp_val = 0.0
        step_size = 2**(- frac_bits)
        while (exp_val + step_size) < 0.1:
            exp_val += step_size
        if (0.1 - exp_val) > (exp_val + step_size - 0.1):
            exp_val += step_size
        exp_val += 1
        assert as_decimal(x, **params) == exp_val


def test_overflow_saturate():
    params = {
        "int_bits": 3,
        "frac_bits": 3,
        "overflow": 0,
        "rounding": 0,
    }
    # 3 int bits and 3 frac bits allows signed numbers in range [-8, 7.875]
    assert as_decimal(create(21.9, **params), **params) == 7.875


def test_overflow_wrap():
    params = {
        "sign_bit": 0,
        "int_bits": 3,
        "frac_bits": 3,
        "overflow": 1,  # OFL_WRAP
        "rounding": 0,  # RND_TRUNCATE
    }
    x = create(21.9, **params)
    # 21.9_10 -> 175_10 after scaling and truncation (multiply by 2**3)
    #         -> 10101111_2
    #         ->   101111_2 (wrap overflow)
    #         -> 5.875_10 (Q3.3 conversion back to decimal)
    assert as_decimal(x, **params) == 5.875


def test_negative_rep():
    assert as_decimal(create(-3.0)) == -3.0


def test_negative_frac():
    params = {
        "sign_bit": 1,
        "int_bits": 5,
        "frac_bits": 10,
    }
    x = create(-4.7, **params)
    # -4.7_10 -> -4812.8_10 after scaling (multiply by 2**10)
    #         -> -4812_10 after truncation
    #         -> 10000000000000000001001011001100_2 (assume 32bit storage)
    #         -> -4.69921875_10 (Q5.10 conversion back to decimal)
    assert as_decimal(x, **params) == -4.69921875


def test_overflow_neg_saturate():
    params = {
        "int_bits": 3,
        "frac_bits": 3,
        "overflow": 0,
        "rounding": 0,
    }
    assert as_decimal(create(-21.9, **params), **params) == -8.0


def test_addition():
    x = create(4)
    y = create(10)
    assert as_decimal(scale(x + y)) == 14.0


def test_overflow_saturated_addition():
    params = {
        "int_bits": 5,
        "frac_bits": 10,
        "overflow": 0,
    }
    x = create(24.567, **params)
    y = create(10, **params)
    assert as_decimal(scale(x + y), **params) == 31.9990234375


def test_rounding_truncated_addition():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 0,
    }
    scaleparams = {
        "in_int_bits": 5,
        "out_int_bits": 5,
        "in_frac_bits": 3,
        "out_frac_bits": 3,
        "rounding": 0,
    }
    # with 3 fractional bits
    x = create(14.567, **params)  # --> 116.536 --> 116 after truncation
    y = create(10, **params)  # --> 80 (after truncation)
    # 196_10 --> 11000.100_2 -> 24.5_10 Q5.3 (.125 * 4 for frac)
    assert as_decimal(scale(x + y, **scaleparams), **params) == 24.5


def test_rounding_nearest_addition():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 1,  # RND_NEAREST
    }
    scaleparams = {
        "in_int_bits": 5,
        "out_int_bits": 5,
        "in_frac_bits": 3,
        "out_frac_bits": 3,
        "rounding": 1,
    }
    # with 3 fractional bits
    x = create(14.567, **params)  # --> 116.536 --> 117 after round nearest
    y = create(10, **params)  # --> 80 (after round nearest)
    # 197_10 --> 11000.101_2 -> 24.625_10 Q5.3 (.125 * 5 for frac)
    assert as_decimal(scale(x + y, **scaleparams), **params) == 24.625


def test_subtraction():
    x = create(10)
    y = create(4)
    assert as_decimal(scale(x - y)) == 6.0


def test_overflow_saturated_subtraction():
    params = {
        "int_bits": 5,
        "frac_bits": 10,
        "overflow": 0,
    }
    x = create(24.567, **params)
    y = create(-10, **params)
    assert as_decimal(scale(x - y), **params) == 31.9990234375


def test_rounding_truncated_subtraction():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 0,
    }
    scaleparams = {
        "in_int_bits": 5,
        "out_int_bits": 5,
        "in_frac_bits": 3,
        "out_frac_bits": 3,
        "rounding": 0,
    }
    # with 3 fractional bits
    x = create(14.567, **params)  # --> 116.536 --> 116 after truncation
    y = create(10, **params)  # --> 80 (after truncation)
    # 36_10 --> 00100.100_2 -> 4.5_10 Q5.3 (.125 * 4 for frac)
    assert as_decimal(scale(x - y, **scaleparams), **params) == 4.5


def test_rounding_nearest_subtraction():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 1,  # RND_NEAREST
    }
    scaleparams = {
        "in_int_bits": 5,
        "out_int_bits": 5,
        "in_frac_bits": 3,
        "out_frac_bits": 3,
        "rounding": 1,
    }
    # with 3 fractional bits
    x = create(14.567, **params)  # --> 116.536 --> 117 after round nearest
    y = create(10, **params)  # --> 80 (after round nearest)
    # 37_10 --> 10100.101_2 -> 4.625_10 Q5.3 (.125 * 5 for frac)
    assert as_decimal(scale(x - y, **scaleparams), **params) == 4.625


def test_multiplication():
    x = create(2)
    y = create(4)
    scaleparams = {
        "in_int_bits": 10,
        "out_int_bits": 5,
        "in_frac_bits": 20,
        "out_frac_bits": 10,
        "rounding": 0,
    }
    assert as_decimal(scale(x * y, **scaleparams)) == 8


def test_overflow_saturated_multiplication():
    params = {
        "int_bits": 5,
        "frac_bits": 10,
        "overflow": 0,
    }
    scaleparams = {
        "in_int_bits": 10,
        "out_int_bits": 5,
        "in_frac_bits": 20,
        "out_frac_bits": 10,
        "rounding": 0,
    }
    x = create(24.567, **params)
    y = create(10, **params)
    assert as_decimal(scale(x * y, **scaleparams), **params) == 31.9990234375


def test_rounding_truncated_multiplication():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 0,
    }
    scaleparams = {
        "in_int_bits": 10,
        "out_int_bits": 5,
        "in_frac_bits": 6,
        "out_frac_bits": 3,
        "rounding": 0,
    }
    # with 3 fractional bits
    x = create(1.567, **params)  # --> 12.536 --> 12 after truncation
    y = create(3.2, **params)  # --> 25.6 --> 25 (after truncation)
    # 300_10 --> 100.101100_2 (pre rescale) --> 100.101_2 (rescaled)
    #        --> 4.625_10 Q5.3 (.125 * 5 for frac)
    assert as_decimal(scale(x * y, **scaleparams), **params) == 4.625


def test_rounding_nearest_multiplication():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 1,  # RND_NEAREST
    }
    scaleparams = {
        "in_int_bits": 10,
        "out_int_bits": 5,
        "in_frac_bits": 6,
        "out_frac_bits": 3,
        "rounding": 1,
    }
    # with 3 fractional bits
    x = create(1.567, **params)  # --> 12.536 --> 13 after round nearest
    y = create(3.2, **params)  # --> 25.6 --> 26 (after round nearest)
    # 338_10 --> 0000000101.010010_2 (pre rescale)
    #        --> 00101.010_2 (rescaled round near)
    #        --> 5.25_10 Q5.3 (.125 * 2 for frac)
    assert as_decimal(scale(x * y, **scaleparams), **params) == 5.25


def test_division():
    x = create(17)
    y = create(4)
    scaleparams = {
        "in_int_bits": 5,
        "out_int_bits": 10,
        "in_frac_bits": 10,
        "out_frac_bits": 20,
        "rounding": 0,
    }
    # for division we need to shift the numerator to output scale before
    # we do the integer divide
    assert as_decimal(scale(x, **scaleparams) / y) == 4.25


def test_rounding_truncated_division():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 0,
    }
    scaleparams = {
        "in_int_bits": 5,
        "out_int_bits": 10,
        "in_frac_bits": 3,
        "out_frac_bits": 6,
        "rounding": 0,
    }
    # with 3 fractional bits
    x = create(10.567, **params)  # --> 84.536 --> 84 after truncation
    y = create(3.2, **params)  # --> 25.6 --> 25 (after truncation)
    # 84_10 --> 672_10 (after pre-scaling) / 25_10
    # 672 / 25 --> 26.88_10 (division)
    #          --> 26_10 (truncated) 
    #          --> 011.010_2
    #          --> 3.25_10 Q5.2 (.125 * 2 for frac)
    assert as_decimal(scale(x, **scaleparams) / y, **params) == 3.25


def test_rounding_nearest_division():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 1,  # RND_NEAREST
    }
    scaleparams = {
        "in_int_bits": 5,
        "out_int_bits": 10,
        "in_frac_bits": 3,
        "out_frac_bits": 6,
        "rounding": 1,
    }
    # with 3 fractional bits
    x = create(10.567, **params)  # --> 84.536 --> 85 after round nearest
    y = create(3.2, **params)  # --> 25.6 --> 26 (after round nearest)
    # 85_10 --> 680_10 (after pre-scaling) / 26_10
    # 680 / 26 --> 26.153846153_10 (division)
    #          --> 26_10 (round to nearest)
    #          --> 011.010_2
    #          --> 3.25_10 Q5.2 (.125 * 2 for frac)
    assert as_decimal(scale(x, **scaleparams) / y, **params) == 3.25


def test_basic_matmatmul():
    params = DEFAULT_FP_VALS.copy()
    params.update({
        "int_bits": 5,
        "frac_bits": 3,
    })
    A = np.array([[create(1.0, **params), create(2.0, **params)],
                  [create(3.0, **params), create(4.0, **params)]],
                 dtype=np.int64)
    B = np.copy(A, order="F")
    out = np.empty([2, 2], dtype=np.int64)
    naive_dot(A, B, out, **params)
    exp_res = np.array([[7.0, 10.0],
                        [15.0, 22.0]])
    assert_tensor_equal(as_decimal(out, **params), exp_res)
