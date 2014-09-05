#!/usr/bin/env python

import sys
import timeit

from mylearn.util.compat import CUDA_GPU
from mylearn.util.error import TooSlowToImplementError


def bench_mat_mat_multiply(backend, classname, A_dims, B_dims, number=10000,
                           repeat=3):
    """
    Generates two random matrices of specified shape utilizing the supplied
    backend, then times how long it takes to multiply them together.

    Arguments:
        backend (str): The dotted module path of the underlying backend
        classname (str): The name of the Backend child class importable from
                         the module desribed in `backend`
        A_dims (list): tuple of positive integers specifying the dimesnions of
                       the left-hand side matrix operand.
        B_dims (list): tuple of positive integers specifying the dimesnions of
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
             "A = %sTensor(np.random.rand(*%s))\n"
             "B = %sTensor(np.random.rand(*%s))\n"
             "out = %sTensor(np.empty([%d, %d]))\n" %
             (backend, classname, classname, classname, classname, str(A_dims),
              classname, str(B_dims), classname, A_dims[0], B_dims[1]))
    try:
        res = timeit.repeat('be.dot(A, B, out)', setup=setup, number=number,
                            repeat=repeat)
    except (NotImplementedError, AttributeError, TooSlowToImplementError):
        res = [float('NaN'), ]
    return min(res)

if __name__ == '__main__':
    number = 100
    repeat = 3
    test_backends = [('mylearn.backends._numpy', 'Numpy'),
                     #  ('mylearn.backends.fixedpoint', 'FixedPoint')]
                     ]
    if CUDA_GPU:
        test_backends.insert(1, ('mylearn.backends._cudamat', 'Cudamat'))
    for A_dims, B_dims in [((2, 2), (2, 2)), ((32, 32), (32, 32)),
                           ((500, 500), (500, 500)),
                           ((1000, 1600), (1600, 1000))]:
        for backend, classname in test_backends:
            sys.stdout.write("%s\t%dx%d dot %dx%d\t%d loop, best of %d:" %
                             (classname, A_dims[0], A_dims[1], B_dims[0],
                              B_dims[1], number, repeat))
            sys.stdout.flush()
            sys.stdout.write("\t%f sec\n" % bench_mat_mat_multiply(backend,
                             classname, A_dims, B_dims, number, repeat))
