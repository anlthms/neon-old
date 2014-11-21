#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

import sys
import timeit

from neon.util.compat import CUDA_GPU
from neon.util.error import TooSlowToImplementError


def bench_mat_mat_multiply(backend, classname, a_dims, b_dims, number=10000,
                           repeat=3):
    """
    Generates two random matrices of specified shape utilizing the supplied
    backend, then times how long it takes to multiply them together.

    Arguments:
        backend (str): The dotted module path of the underlying backend
        classname (str): The name of the Backend child class importable from
                         the module desribed in `backend`
        a_dims (list): tuple of positive integers specifying the dimesnions of
                       the left-hand side matrix operand.
        b_dims (list): tuple of positive integers specifying the dimesnions of
                       the right-hand side matrix operand.
        number (int, optional): The number of times to perform the operation
                                so that per loop time can be reported.
                                Defaults to 10000
        repeat (int, optional): The number of times to repeat the timing
                                experiment.  Defaults to 3.

    Returns:
        float: min elapsed time (in seconds) of the repeated runs.
    """
    setup = ("import numpy as np\n" "from %s import %s, %sTensor\n"
             "be = %s(rng_seed=0)\n"
             "a = %sTensor(np.random.rand(*%s))\n"
             "b = %sTensor(np.random.rand(*%s))\n"
             "out = %sTensor(np.empty([%d, %d]))\n" %
             (backend, classname, classname, classname, classname, str(a_dims),
              classname, str(b_dims), classname, a_dims[0], b_dims[1]))
    try:
        res = timeit.repeat('be.dot(a, b, out)', setup=setup, number=number,
                            repeat=repeat)
    except (NotImplementedError, AttributeError, TooSlowToImplementError):
        res = [float('NaN'), ]
    return min(res)

if __name__ == '__main__':
    number = 100
    repeat = 3
    test_backends = [('neon.backends.cpu', 'CPU'),
                     ('neon.backends.unsupported._numpy', 'Numpy64'),
                     #  ('neon.backends.fixedpoint', 'FixedPoint')]
                     ]
    if CUDA_GPU:
        # TODO: once cudanet init/shutdown resolved replace:
        test_backends.insert(1, ('neon.backends.unsupported._cudamat',
                                 'Cudamat'))
        # with:
        # test_backends.insert(1, ('neon.backends.gpu', 'GPU'))
    for a_dims, b_dims in [((2, 2), (2, 2)), ((32, 32), (32, 32)),
                           ((500, 500), (500, 500)),
                           ((1000, 1600), (1600, 1000))]:
        for backend, classname in test_backends:
            sys.stdout.write("%s\t%dx%d dot %dx%d\t%d loop, best of %d:" %
                             (classname, a_dims[0], a_dims[1], b_dims[0],
                              b_dims[1], number, repeat))
            sys.stdout.flush()
            sys.stdout.write("\t%f sec\n" % bench_mat_mat_multiply(backend,
                             classname, a_dims, b_dims, number, repeat))
