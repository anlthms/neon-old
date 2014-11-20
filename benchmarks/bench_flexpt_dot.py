#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
import sys
import timeit

SIZE = 500
NUMBER = 10
REPEAT = 3


def test_flexpt_dtype():
    setup = ("import numpy as np\n"
             "from neon.backends.flexpoint "
             "import Flexpoint, FlexpointTensor\n"
             "A = FlexpointTensor(np.random.randn(%d, %d))\n"
             "B = FlexpointTensor(np.random.randn(%d, %d))\n"
             "out = Flexpoint.zeros([%d, %d])" %
             (SIZE, SIZE, SIZE, SIZE, SIZE, SIZE))
    res = timeit.repeat("Flexpoint.dot(A, B, out)", setup=setup,
                        number=NUMBER, repeat=REPEAT)
    return min(res)


def test_float64_np():
    setup = ("import numpy as np\n"
             "A = np.random.randn(%d, %d)\n"
             "B = np.random.randn(%d, %d)\n"
             "out = np.empty([%d, %d])" %
             (SIZE, SIZE, SIZE, SIZE, SIZE, SIZE))
    res = timeit.repeat("np.dot(A, B, out)", setup=setup,
                        number=NUMBER, repeat=REPEAT)
    return min(res)


def test_float32_np():
    setup = ("import numpy as np\n"
             "A = np.random.randn(%d, %d)\n"
             "B = np.random.randn(%d, %d)\n"
             "A = A.astype(np.float32)\n"
             "B = B.astype(np.float32)\n"
             "out = np.empty([%d, %d], np.float32)" %
             (SIZE, SIZE, SIZE, SIZE, SIZE, SIZE))
    res = timeit.repeat("np.dot(A, B, out)", setup=setup,
                        number=NUMBER, repeat=REPEAT)
    return min(res)


def test_int64_np():
    setup = ("import numpy as np\n"
             "A = np.random.randint(10, size=[%d, %d])\n"
             "B = np.random.randint(10, size=[%d, %d])\n"
             "out = np.empty([%d, %d], np.int64)" %
             (SIZE, SIZE, SIZE, SIZE, SIZE, SIZE))
    res = timeit.repeat("np.dot(A, B, out)", setup=setup,
                        number=NUMBER, repeat=REPEAT)
    return min(res)


def test_int32_np():
    setup = ("import numpy as np\n"
             "A = np.random.randint(10, size=[%d, %d])\n"
             "B = np.random.randint(10, size=[%d, %d])\n"
             "A = A.astype(np.int32)\n"
             "B = B.astype(np.int32)\n"
             "out = np.empty([%d, %d], np.int32)" %
             (SIZE, SIZE, SIZE, SIZE, SIZE, SIZE))
    res = timeit.repeat("np.dot(A, B, out)", setup=setup,
                        number=NUMBER, repeat=REPEAT)
    return min(res)


def test_flexpt_cython():
    setup = ("import numpy as np\n"
             "from neon.backends.flexpt_cython import (naive_dot,"
             "                                         flexpt_dtype)\n"
             "dtype = flexpt_dtype(1, 5, 10, 0, 0)\n"
             "A = np.random.randint(10, size=[%d, %d])\n"
             "B = np.random.randint(10, size=[%d, %d])\n"
             "A = A.astype(np.int64)\n"
             "B = B.astype(np.int64)\n"
             "out = np.empty([%d, %d], np.int64)" %
             (SIZE, SIZE, SIZE, SIZE, SIZE, SIZE))
    res = timeit.repeat("naive_dot(A, B.transpose(), out, dtype, dtype, "
                        "dtype)", setup=setup, number=NUMBER, repeat=REPEAT)
    return min(res)

if __name__ == '__main__':
    for test in [test_float64_np, test_float32_np, test_int64_np,
                 test_int32_np, test_flexpt_cython]:
        print("%s\t%dx%d dot\t%d loop, min of %d:\t%f" %
              (test.__name__, SIZE, SIZE, NUMBER, REPEAT, test()))
        sys.stdout.flush()
