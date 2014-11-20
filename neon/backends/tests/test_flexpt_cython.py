#!/usr/bin/env/python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np

from neon.backends.flexpt_cython import (flex_from_float, flex_to_float,
                                         flex_from_float_array, elemfloat,
                                         flex_to_float_array, fp_rescale,
                                         naive_dot, flexpt_dtype, elemtype)
from neon.util.testing import assert_tensor_equal


def test_creation_and_rep():
    dtype = flexpt_dtype(sign_bit=False, int_bits=3, frac_bits=9, overflow=0,
                         rounding=0)
    internal_val = flex_from_float(1.0, dtype)
    assert internal_val == 512  # 512 == 2**9
    assert flex_to_float(internal_val, dtype) == 1.0


def test_fracbit_rep():
    dtype = flexpt_dtype(sign_bit=True, int_bits=1, frac_bits=0, overflow=0,
                         rounding=1)
    for frac_bits in range(0, 6):
        dtype["frac_bits"] = frac_bits
        x = flex_from_float(1.1, dtype)
        exp_val = 0.0
        step_size = 2**(- frac_bits)
        while (exp_val + step_size) < 0.1:
            exp_val += step_size
        if (0.1 - exp_val) > (exp_val + step_size - 0.1):
            exp_val += step_size
        exp_val += 1
        assert flex_to_float(x, dtype) == exp_val


def test_overflow_saturate():
    dtype = flexpt_dtype(sign_bit=True, int_bits=3, frac_bits=3, overflow=0,
                         rounding=0)
    # 3 int bits and 3 frac bits allows signed numbers in range [-8, 7.875]
    assert flex_to_float(flex_from_float(21.9, dtype), dtype) == 7.875


def test_overflow_wrap():
    dtype = flexpt_dtype(sign_bit=False, int_bits=3, frac_bits=3, overflow=1,
                         rounding=0)
    x = flex_from_float(21.9, dtype)
    # 21.9_10 -> 175_10 after scaling and truncation (multiply by 2**3)
    #         -> 10101111_2
    #         ->   101111_2 (wrap overflow)
    #         -> 47_10 (as decimal)
    #         -> 5.875_10 (Q3.3 conversion back to decimal)
    assert x == 47
    assert flex_to_float(x, dtype) == 5.875


def test_underflow_saturate():
    dtype = flexpt_dtype(sign_bit=True, int_bits=3, frac_bits=3, overflow=0,
                         rounding=0)
    # 3 frac bits gives minimal precision of 2**-3 == 0.125
    assert flex_to_float(flex_from_float(0.05, dtype), dtype) == 0.0


def test_underflow_wrap():
    dtype = flexpt_dtype(sign_bit=False, int_bits=3, frac_bits=3, overflow=1,
                         rounding=0)
    # 3 frac bits gives minimal precision of 2**-3 == 0.125
    # wrapping shouldn't have any impact here
    assert flex_to_float(flex_from_float(0.05, dtype), dtype) == 0.0


def test_midpoint_truncation():
    dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0, overflow=0,
                         rounding=0)
    x = flex_from_float(2.5, dtype)
    # 2.5_10 -> 2_10 after scaling and truncation
    assert x == 2
    assert flex_to_float(x, dtype) == 2.0


def test_midpoint_round_nearest():
    dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0, overflow=0,
                         rounding=1)
    x = flex_from_float(2.5, dtype)
    # 2.5_10 -> 3_10 after scaling and round to nearest (biased or unbiased)
    assert x == 3
    assert flex_to_float(x, dtype) == 3.0


def test_midpoint_rescale_truncation():
    in_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=2,
                            overflow=0, rounding=0)
    out_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0,
                             overflow=0, rounding=0)
    x = flex_from_float(2.5, in_dtype)
    # 2.5_10 -> 10_10 after scaling and truncation (multiply by 2**2)
    #        -> 00010.10_2
    assert x == 10
    #        -> 00010_2 truncating during rescale
    #        -> 2_10
    x = fp_rescale(x, in_dtype, out_dtype)
    assert x == 2
    assert flex_to_float(x, out_dtype) == 2.0


def test_midpoint_rescale_round_nearest():
    in_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=2,
                            overflow=0, rounding=0)
    out_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0,
                             overflow=0, rounding=1)
    x = flex_from_float(2.5, in_dtype)
    # 2.5_10 -> 10_10 after scaling and truncation (multiply by 2**2)
    #        -> 00010.10_2
    assert x == 10
    #        -> 00011_2 round nearest (i.e add 0.5_10 then truncate)
    #        -> 3_10
    x = fp_rescale(x, in_dtype, out_dtype)
    assert x == 3
    assert flex_to_float(x, out_dtype) == 3.0


def test_point_shift_right():
    dtype = flexpt_dtype(sign_bit=True, int_bits=0, frac_bits=3, overflow=0,
                         rounding=0, point_shift=3)
    # .03125_10 -> 0.25_10 after scaling with frac_bits alone -> 0 when rounded
    #           -> 2_10 with 3 additional point shift bits -> .010_2
    assert flex_to_float(flex_from_float(0.03125, dtype), dtype) == 0.03125


def test_point_shift_left():
    dtype = flexpt_dtype(sign_bit=True, int_bits=3, frac_bits=0, overflow=0,
                         rounding=0, point_shift=-3)
    # 32_10 -> 100000_2 in binary, which won't fit with 3 int bits and 0 frac
    #       -> 100xxx._2 with 3 point shift bits
    assert flex_to_float(flex_from_float(32.0, dtype), dtype) == 32.0


def test_negative_rep():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    assert flex_to_float(flex_from_float(-3.0, dtype), dtype) == -3.0


def test_negative_frac():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(-4.7, dtype)
    # -4.7_10 -> -4812.8_10 after scaling (multiply by 2**10)
    #         -> -4812_10 after truncation
    #         -> 10000000000000000001001011001100_2 (assume 32bit storage)
    #         -> -4.69921875_10 (Q5.10 conversion back to decimal)
    assert x == -4812
    assert flex_to_float(x, dtype) == -4.69921875


def test_overflow_neg_saturate():
    dtype = flexpt_dtype(sign_bit=True, int_bits=3, frac_bits=3, overflow=0,
                         rounding=0)
    assert flex_to_float(flex_from_float(-21.9, dtype), dtype) == -8.0


def test_overflow_neg_wrap():
    dtype = flexpt_dtype(sign_bit=False, int_bits=3, frac_bits=3, overflow=1,
                         rounding=0)
    x = flex_from_float(-21.9, dtype)
    # -21.9_10 -> -175_10 after scaling and truncation (multiply by 2**3)
    #         -> 1...01010001_2
    #         ->       010001_2 (wrap overflow)
    #         -> 17_10 (as decimal)
    #         -> 2.125_10 (Q3.3 conversion back to decimal)
    assert x == 17
    assert flex_to_float(x, dtype) == 2.125


def test_underflow_neg_saturate():
    dtype = flexpt_dtype(sign_bit=True, int_bits=3, frac_bits=3, overflow=0,
                         rounding=0)
    # 3 frac bits gives minimal precision of 2**-3 == 0.125
    assert flex_to_float(flex_from_float(-0.05, dtype), dtype) == 0.0


def test_underflow_neg_wrap():
    dtype = flexpt_dtype(sign_bit=False, int_bits=3, frac_bits=3, overflow=1,
                         rounding=0)
    # 3 frac bits gives minimal precision of 2**-3 == 0.125
    # wrapping shouldn't have any impact here
    assert flex_to_float(flex_from_float(-0.05, dtype), dtype) == 0.0


def test_midpoint_neg_truncation():
    dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0, overflow=0,
                         rounding=0)
    x = flex_from_float(-2.5, dtype)
    # -2.5_10 -> -2_10 after scaling and truncation
    assert x == -2
    assert flex_to_float(x, dtype) == -2.0


# TODO: test negative overflow saturation during rescale!!!


def test_midpoint_neg_rescale_truncation():
    in_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=2,
                            overflow=0, rounding=0)
    out_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0,
                             overflow=0, rounding=0)
    x = flex_from_float(-2.5, in_dtype)
    # -2.5_10 -> -10_10 after scaling and truncation (multiply by 2**2)
    #         -> 11101.10_2
    assert x == -10
    #        -> 11101_2 truncating during rescale
    #        -> -3_10
    x = fp_rescale(x, in_dtype, out_dtype)
    assert x == -3
    assert flex_to_float(x, out_dtype) == -3.0


def test_midpoint_neg_round_nearest():
    dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0, overflow=0,
                         rounding=1)
    x = flex_from_float(-2.5, dtype)
    # -2.5_10 -> -2_10 after scaling and (biased) round to nearest
    assert x == -2
    assert flex_to_float(x, dtype) == -2.0


def test_midpoint_neg_round_nearest_unbiased():
    dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0, overflow=0,
                         rounding=2)
    x = flex_from_float(-2.5, dtype)
    # -2.5_10 -> -3_10 after scaling and (unbiased) round to nearest
    assert x == -3
    assert flex_to_float(x, dtype) == -3.0


def test_midpoint_neg_rescale_round_nearest():
    in_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=2,
                            overflow=0, rounding=0)
    out_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0,
                             overflow=0, rounding=1)
    x = flex_from_float(-2.5, in_dtype)
    # -2.5_10 -> -10_10 after scaling and truncation (multiply by 2**2)
    #         -> 11101.10_2
    assert x == -10
    # biased round nearest implies adding 0.5 and truncating
    #        -> truncate(11101.10_2 + 00000.10_2)
    #        -> 11110_2
    #        -> -2_10
    x = fp_rescale(x, in_dtype, out_dtype)
    assert x == -2
    assert flex_to_float(x, out_dtype) == -2.0


def test_midpoint_neg_rescale_round_nearest_unbiased():
    in_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=2,
                            overflow=0, rounding=0)
    out_dtype = flexpt_dtype(sign_bit=False, int_bits=5, frac_bits=0,
                             overflow=0, rounding=2)
    x = flex_from_float(-2.5, in_dtype)
    # -2.5_10 -> -10_10 after scaling and truncation (multiply by 2**2)
    #         -> 11101.10_2
    assert x == -10
    # unbiased round nearest implies not adding 0.5 in the rounding for
    # negative midpoints (separate check required)
    #        -> truncate(11101.10_2 + 00000.10_2)
    #        -> 11110_2
    #        -> -2_10
    #        -> 00011_2 round nearest (i.e add 0.5_10 then truncate)
    #        -> -3_10
    x = fp_rescale(x, in_dtype, out_dtype)
    assert x == -3
    assert flex_to_float(x, out_dtype) == -3.0


def test_addition():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(4, dtype)
    y = flex_from_float(10, dtype)
    assert flex_to_float(fp_rescale(x + y, dtype, dtype), dtype) == 14.0


def test_neg_addition():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(-4, dtype)
    y = flex_from_float(-10, dtype)
    assert flex_to_float(fp_rescale(x + y, dtype, dtype), dtype) == -14.0


def test_mixed_addition():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(4, dtype)
    y = flex_from_float(-10, dtype)
    assert flex_to_float(fp_rescale(x + y, dtype, dtype), dtype) == -6.0


def test_overflow_saturated_addition():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(24.567, dtype)
    y = flex_from_float(10, dtype)
    assert flex_to_float(fp_rescale(x + y, dtype, dtype),
                         dtype) == 31.9990234375


def test_rounding_truncated_addition():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=0)
    x = flex_from_float(14.567, dtype)  # --> 116.536 --> 116 after truncation
    assert x == 116
    y = flex_from_float(10, dtype)  # --> 80 (after truncation)
    assert y == 80
    # 196_10 --> 11000.100_2 -> 24.5_10 Q5.3 (.125 * 4 for frac)
    assert flex_to_float(fp_rescale(x + y, dtype, dtype), dtype) == 24.5


def test_rounding_nearest_addition():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=1)
    x = flex_from_float(14.567, dtype)  # --> 116.536 --> 117 after round near
    assert x == 117
    y = flex_from_float(10, dtype)  # --> 80 (after round nearest)
    assert y == 80
    # 197_10 --> 11000.101_2 -> 24.625_10 Q5.3 (.125 * 5 for frac)
    assert flex_to_float(fp_rescale(x + y, dtype, dtype), dtype) == 24.625


def test_mixed_dtype_addition():
    x_dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                           rounding=1)
    y_dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=5, overflow=0,
                           rounding=1)
    out_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=5,
                             overflow=0, rounding=1)
    x = flex_from_float(14.567, x_dtype)  # --> 116.536 --> 117 after round
    assert x == 117
    y = flex_from_float(10, y_dtype)  # --> 320 (after round nearest)
    assert y == 320
    scale_y = fp_rescale(y, y_dtype, x_dtype)  # --> 80 (after match scale)
    # 197_10 --> 788_10 (after output scaling)
    #        --> 11000.10100_2 -> 24.625_10 Q10.5 (.03125 * 20 for frac)
    assert flex_to_float(fp_rescale(x + scale_y, x_dtype, out_dtype),
                         out_dtype) == 24.625


def test_subtraction():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(10, dtype)
    y = flex_from_float(4, dtype)
    assert flex_to_float(fp_rescale(x - y, dtype, dtype), dtype) == 6.0


def test_overflow_saturated_subtraction():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(24.567, dtype)
    y = flex_from_float(-10, dtype)
    assert flex_to_float(fp_rescale(x - y, dtype, dtype),
                         dtype) == 31.9990234375


def test_rounding_truncated_subtraction():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=0)
    x = flex_from_float(14.567, dtype)  # --> 116.536 --> 116 after truncation
    assert x == 116
    y = flex_from_float(10, dtype)  # --> 80 (after truncation)
    assert y == 80
    res = fp_rescale(x - y, dtype, dtype)
    # 36_10 --> 00100.100_2 -> 4.5_10 Q5.3 (.125 * 4 for frac)
    assert res == 36
    assert flex_to_float(res, dtype) == 4.5


def test_rounding_nearest_subtraction():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=1)
    x = flex_from_float(14.567, dtype)  # --> 116.536 --> 117 after round near
    assert x == 117
    y = flex_from_float(10, dtype)  # --> 80 (after round nearest)
    assert y == 80
    res = fp_rescale(x - y, dtype, dtype)
    # 37_10 --> 10100.101_2 -> 4.625_10 Q5.3 (.125 * 5 for frac)
    assert res == 37
    assert flex_to_float(res, dtype) == 4.625


def test_multiplication():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(2, dtype)  # --> 2048_10
    assert x == 2048
    y = flex_from_float(4, dtype)  # --> 4096_10
    assert y == 4096
    # before overflow and round checking, the integer multiplication yields
    # a result in Q(x.int_bits + y.int_bits).(x.frac_bits + y.frac_bits) format
    mul_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=20,
                             overflow=0, rounding=0)
    res = fp_rescale(x * y, mul_dtype, dtype)
    # 2048 * 4096 --> 8388608_10 -> 1000.00000000000000000000_2 (Q10.20 format)
    #             -> 1000.0000000000_2 (after cast back to Q5.10 format)
    #             -> 8.0_10
    assert flex_to_float(res, dtype) == 8


def test_overflow_saturated_multiplication():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(24.567, dtype)
    y = flex_from_float(10, dtype)
    mul_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=20,
                             overflow=0, rounding=0)
    res = fp_rescale(x * y, mul_dtype, dtype)
    assert flex_to_float(res, dtype) == 31.9990234375


def test_rounding_truncated_multiplication():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=0)
    x = flex_from_float(1.567, dtype)  # --> 12.536 --> 12 after truncation
    assert x == 12
    y = flex_from_float(3.2, dtype)  # --> 25.6 --> 25 (after truncation)
    assert y == 25
    mul_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=6,
                             overflow=0, rounding=0)
    res = fp_rescale(x * y, mul_dtype, dtype)
    # 300_10 --> 100.101100_2 (pre rescale)
    #        --> 37_10 --> 100.101_2 (rescaled)
    #        --> 4.625_10 Q5.3 (.125 * 5 for frac)
    assert res == 37
    assert flex_to_float(res, dtype) == 4.625


def test_rounding_nearest_multiplication():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=1)
    x = flex_from_float(1.567, dtype)  # --> 12.536 --> 13 after round nearest
    assert x == 13
    y = flex_from_float(3.2, dtype)  # --> 25.6 --> 26 (after round nearest)
    assert y == 26
    mul_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=6,
                             overflow=0, rounding=1)
    res = fp_rescale(x * y, mul_dtype, dtype)
    # 338_10 --> 0000000101.010010_2 (pre rescale)
    #        --> 42_10 --> 00101.010_2 (rescaled round near)
    #        --> 5.25_10 Q5.3 (.125 * 2 for frac)
    assert res == 42
    assert flex_to_float(res, dtype) == 5.25


def test_division():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=10, overflow=0,
                         rounding=0)
    x = flex_from_float(17, dtype)
    y = flex_from_float(4, dtype)
    # for division we need to shift the numerator to output scale before
    # we do the integer divide.  If x has m fractional bits, and y has n
    # fractional bits, and we want the output to have f fractional bits,
    # we have to left shift the numerator by f - (m - n) total bits
    # in this example f=m=n == 10, so we need to first left shift by 10
    num = x << 10
    assert flex_to_float(num / y, dtype) == 4.25


def test_rounding_truncated_division():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=0)
    x = flex_from_float(10.567, dtype)  # --> 84.536 --> 84 after truncation
    assert x == 84
    y = flex_from_float(3.2, dtype)  # --> 25.6 --> 25 (after truncation)
    assert y == 25
    # 84_10 --> 672_10 (after pre-scaling) / 25_10
    # 672 / 25 --> 26.88_10 (division)
    #          --> 26_10 (truncated)
    #          --> 011.010_2
    #          --> 3.25_10 Q5.3 (.125 * 2 for frac)
    div_num_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=6,
                                 overflow=0, rounding=0)
    num = fp_rescale(x, dtype, div_num_dtype)
    assert num == 672
    assert flex_to_float(num / y, dtype) == 3.25


def test_rounding_nearest_division():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=1)
    x = flex_from_float(10.567, dtype)  # --> 84.536 --> 85 after round near
    assert x == 85
    y = flex_from_float(3.2, dtype)  # --> 25.6 --> 26 (after round nearest)
    assert y == 26
    # 85_10 --> 680_10 (after pre-scaling) / 26_10
    # 680 / 26 --> 26.153846153_10 (division)
    #          --> 26_10 (round to nearest)
    #          --> 011.010_2
    #          --> 3.25_10 Q5.2 (.125 * 2 for frac)
    div_num_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=6,
                                 overflow=0, rounding=1)
    num = fp_rescale(x, dtype, div_num_dtype)
    assert num == 680
    assert flex_to_float(num / y, dtype) == 3.25


def test_mixed_dtype_division():
    x_dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                           rounding=1)
    y_dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=5, overflow=0,
                           rounding=1)
    out_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=5,
                             overflow=0, rounding=0)
    x = flex_from_float(14.567, x_dtype)  # --> 116.536 --> 117 after round
    assert x == 117
    y = flex_from_float(10, y_dtype)  # --> 320 (after round nearest)
    assert y == 320
    num = x << (out_dtype['frac_bits'] - (x_dtype['frac_bits'] -
                                          y_dtype['frac_bits']))
    # 117_10 --> 14976_10 (after pre-scaling numerator 5 - (3 - 5) == 7 bits
    assert num == 14976
    # 14976 / 320 --> 46.8_10 (division)
    #             --> 46_10 (truncation)
    #             --> 1.01110_2
    #             --> 1.4375_10 Q10.5 (.03125 * 14 for frac)
    assert flex_to_float(num / y, out_dtype) == 1.4375


def test_basic_matmatmul():
    dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                         rounding=0)
    a = flex_from_float_array(np.array([[1.0, 2.0], [3.0, 4.0]], elemfloat),
                              dtype)
    b = np.copy(a, order="F")
    out = np.empty([2, 2], dtype=elemtype)
    naive_dot(a, b, out, dtype, dtype, dtype)
    exp_res = np.array([[7.0, 10.0], [15.0, 22.0]], elemfloat)
    assert_tensor_equal(flex_to_float_array(out, dtype), exp_res)


def test_mixed_dtype_matmatmul():
    a_dtype = flexpt_dtype(sign_bit=True, int_bits=5, frac_bits=3, overflow=0,
                           rounding=0)
    b_dtype = flexpt_dtype(sign_bit=True, int_bits=4, frac_bits=5, overflow=0,
                           rounding=0)
    a = flex_from_float_array(np.array([[1.0, 2.0], [3.0, 4.0]], elemfloat),
                              a_dtype)
    b = flex_from_float_array(np.array([[1.0, 2.0], [3.0, 4.0]], elemfloat),
                              b_dtype)
    b = np.asfortranarray(b)
    out_dtype = flexpt_dtype(sign_bit=True, int_bits=10, frac_bits=5,
                             overflow=0, rounding=0)
    out = np.empty([2, 2], dtype=elemtype)
    naive_dot(a, b, out, a_dtype, b_dtype, out_dtype)
    exp_res = np.array([[7.0, 10.0], [15.0, 22.0]], elemfloat)
    assert_tensor_equal(flex_to_float_array(out, out_dtype), exp_res)
