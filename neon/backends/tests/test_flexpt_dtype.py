#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

from neon.backends import flexpt

# defaults
DEFAULT_FP_VALS = {
    "sign_bit": True,
    "int_bits": 5,
    "frac_bits": 10,
    "overflow": 0,  # 0 == OFL_SATURATE
    "rounding": 0  # 0 == RND_TRUNCATE
}


def check_rep(x, exp_val, **kwargs):
    args = dict(DEFAULT_FP_VALS)
    args.update(kwargs)
    if "sign_bit" in args:
        args["sign_bit"] = int(args["sign_bit"])
    args["sym"] = '+' if exp_val >= 0 else '-'
    args["exp_val"] = abs(exp_val)
    print(repr(x))
    print("flexpt({sym}{exp_val}, sign_bit={sign_bit}, int_bits={int_bits}, "
          "frac_bits={frac_bits}, overflow={overflow}, "
          "rounding={rounding})".format(**args))
    print("")
    assert repr(x) == ("flexpt({sym}{exp_val}, sign_bit={sign_bit}, "
                       "int_bits={int_bits}, frac_bits={frac_bits}, "
                       "overflow={overflow}, rounding={rounding})".
                       format(**args))


def test_creation_and_rep():
    params = {
        "sign_bit": False,
        "int_bits": 3,
        "frac_bits": 9,
        "overflow": 0,
        "rounding": 0,
    }
    x = flexpt(1.0, **params)
    check_rep(x, 1.0, **params)


def test_defaulting():
    x = flexpt(5)
    check_rep(x, 5.0)

    params = {
        "int_bits": 4,
        "sign_bit": 0,
        "overflow": 0,
        "frac_bits": 3,
    }
    x = flexpt(9.0, **params)
    check_rep(x, 9.0, **params)


def test_fracbit_rep():
    params = {
        "sign_bit": True,
        "int_bits": 1,
        "rounding": 1,  # RND_NEAREST
    }
    for frac_bits in range(0, 6):
        params["frac_bits"] = frac_bits
        x = flexpt(1.1, **params)
        exp_val = 0.0
        step_size = 2**(- frac_bits)
        while (exp_val + step_size) < 0.1:
            exp_val += step_size
        if (0.1 - exp_val) > (exp_val + step_size - 0.1):
            exp_val += step_size
        exp_val += 1
        # TODO: figure out a way to make yield work with keyword args
        # yield check_rep, x, exp_val, **params
        check_rep(x, exp_val, **params)


def test_overflow_saturate():
    params = {
        "int_bits": 3,
        "frac_bits": 3,
        "overflow": 0,
        "rounding": 0,
    }
    # 3 int bits and 3 frac bits allows signed numbers in range [-8, 7.875]
    x = flexpt(21.9, **params)
    check_rep(x, 7.875, **params)


def test_overflow_wrap():
    params = {
        "sign_bit": 0,
        "int_bits": 3,
        "frac_bits": 3,
        "overflow": 1,  # OFL_WRAP
        "rounding": 0,  # RND_TRUNCATE
    }
    x = flexpt(21.9, **params)
    # 21.9_10 -> 175_10 after scaling and truncation (multiply by 2**3)
    #         -> 10101111_2
    #         ->   101111_2 (wrap overflow)
    #         -> 5.875_10 (Q3.3 conversion back to decimal)
    check_rep(x, 5.875, **params)


def test_negative_rep():
    x = flexpt(-3.0)
    check_rep(x, -3.0)


def test_negative_frac():
    params = {
        "frac_bits": 10,
    }
    x = flexpt(-4.7, **params)
    check_rep(x, -4.7001953125, **params)


def test_overflow_neg_saturate():
    params = {
        "int_bits": 3,
        "frac_bits": 3,
        "overflow": 0,
        "rounding": 0,
    }
    x = flexpt(-21.9, **params)
    check_rep(x, -8.0, **params)


def test_addition():
    x = flexpt(4)
    y = flexpt(10)
    check_rep(x + y, 14.0)


def test_overflow_saturated_addition():
    params = {
        "int_bits": 5,
        "frac_bits": 10,
        "overflow": 0,
    }
    x = flexpt(24.567, **params)
    y = flexpt(10, **params)
    check_rep(x + y, 31.9990234375, **params)


def test_rounding_truncated_addition():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 0,
    }
    # with 3 fractional bits
    x = flexpt(14.567, **params)  # --> 116.536 --> 116 after truncation
    y = flexpt(10, **params)  # --> 80 (after truncation)
    # 196_10 --> 11000.100_2 -> 24.5_10 Q5.3 (.125 * 4 for frac)
    check_rep(x + y, 24.5, **params)


def test_rounding_nearest_addition():
    params = {
        "int_bits": 5,
        "frac_bits": 3,
        "rounding": 1,  # RND_NEAREST
    }
    # with 3 fractional bits
    x = flexpt(14.567, **params)  # --> 116.536 --> 117 after round nearest
    y = flexpt(10, **params)  # --> 80 (after truncation)
    # 197_10 --> 11000.101_2 -> 24.625_10 Q5.3 (.125 * 5 for frac)
    check_rep(x + y, 24.625, **params)
