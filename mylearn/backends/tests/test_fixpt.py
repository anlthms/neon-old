#!/usr/bin/env/python

from mylearn.backends import fixpt

# defaults
DEFAULT_FP_VALS = {
    "sign_bit": True,
    "int_bits": 5,
    "frac_bits": 10,
    "overflow": "saturate",
    "rounding": "truncate"
}


def check_rep(x, exp_val, **kwargs):
    args = dict(DEFAULT_FP_VALS)
    args.update(kwargs)
    if "sign_bit" in args:
        args["sign_bit"] = int(args["sign_bit"])
    args["sym"] = '+' if exp_val >= 0 else '-'
    args["exp_val"] = abs(exp_val)
    print(repr(x))
    print("fixpt({sym}{exp_val}, sign_bit={sign_bit}, int_bits={int_bits}, "
          "frac_bits={frac_bits}, overflow='{overflow}', "
          "rounding='{rounding}')".format(**args))
    print("")
    assert repr(x) == ("fixpt({sym}{exp_val}, sign_bit={sign_bit}, "
                       "int_bits={int_bits}, frac_bits={frac_bits}, "
                       "overflow='{overflow}', rounding='{rounding}')".
                       format(**args))


def test_creation_and_rep():
    params = {
        "sign_bit": False,
        "int_bits": 3,
        "frac_bits": 9,
        "overflow": 'saturate',
        "rounding": 'truncate',
    }
    x = fixpt(1.0, **params)
    check_rep(x, 1.0, **params)


def test_defaulting():
    x = fixpt(5)
    check_rep(x, 5.0)

    params = {
        "int_bits": 4,
        "sign_bit": 0,
        "overflow": 'saturate',
        "frac_bits": 3,
    }
    x = fixpt(9.0, **params)
    check_rep(x, 9.0, **params)


def test_fraction_rep():
    params = {
        "sign_bit": True,
        "int_bits": 1,
    }
    # for frac_bits in range(0, 6):
    for frac_bits in range(0, 4):
        params["frac_bits"] = frac_bits
        x = fixpt(1.1, **params)
        exp_val = 0.
        step_size = 2**(- frac_bits)
        while (exp_val + step_size) < 0.1:
            exp_val += step_size
        if (0.1 - exp_val) > (exp_val + step_size - 0.1):
            exp_val += step_size
        exp_val += 1
        # TODO: figure out a way to make yield work with keyword args
        # yield check_rep, x, exp_val, **params
        check_rep(x, exp_val, **params)


def test_addition():
    x = fixpt(4)
    y = fixpt(10)
    check_rep(x + y, 14.0)


def test_overflow_addition():
    params = {
        "int_bits": 5,
        "frac_bits": 10,
    }
    x = fixpt(24, **params)
    y = fixpt(10, **params)
    check_rep(x + y, 31.9990234375, **params)
