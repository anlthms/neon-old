#!/usr/bin/env python
"""
Times various indexing approaches (fancy vs. take)
"""

import sys
import timeit

from mylearn.util.compat import CUDA_GPU
from mylearn.util.error import TooSlowToImplementError


def bench_mat_indexing(backend, classname, a_dims, indices, lop, rop,
                       number=10000, repeat=3):
    """
    Generates a random matrix of specified shape utilizing the supplied
    backend, then times how long it takes to index into it using the
    indices passed according to the approach specified via lop and rop.

    Arguments:
        backend (str): The dotted module path of the underlying backend
        classname (str): The name of the Backend child class importable from
                         the module desribed in `backend`
        a_dims (list): tuple of positive integers specifying the dimesnions of
                       the matrix to index.
        indices (array_like): list of positive integers to index into.
        lop (str): How to call the indexing.  Should be something like '.take('
                   or '['
        rop (str): What to put after specifying the indices.  Typically this
                   will be something like ')' or ']'
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
             "A = %sTensor(np.random.rand(*%s))\n" %
             (backend, classname, classname, classname, classname,
              str(a_dims)))
    try:
        res = timeit.repeat('A%s%s%s' % (lop, indices, rop), setup=setup,
                            number=number, repeat=repeat)
    except (NotImplementedError, AttributeError, TooSlowToImplementError):
        res = [float('NaN'), ]
    return min(res)


def bench_mat_slicing(backend, classname, a_dims, slices, axes, number=10000,
                      repeat=3):
    """
    Generates a random matrix of specified shape utilizing the supplied
    backend, then times how long it takes to slice into it using the
    slice and axis passed

    Arguments:
        backend (str): The dotted module path of the underlying backend
        classname (str): The name of the Backend child class importable from
                         the module desribed in `backend`
        a_dims (list): tuple of positive integers specifying the dimesnions of
                       the matrix to index.
        slices (slice): slice object giving contiguous region to index.
        axes (int): dimension along which to slice.
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
             "A = %sTensor(np.random.rand(*%s))\n" %
             (backend, classname, classname, classname, classname,
              str(a_dims)))
    index_str = [':', ] * len(a_dims)
    index_str[axes] = '%d:%d' % (slices.start, slices.stop)
    try:
        res = timeit.repeat('A[%s]' % ', '.join(index_str), setup=setup,
                            number=number, repeat=repeat)
    except (NotImplementedError, AttributeError, TooSlowToImplementError):
        res = [float('NaN'), ]
    return min(res)

if __name__ == '__main__':
    number = 10000
    repeat = 3
    test_backends = [('mylearn.backends._numpy', 'Numpy'), ]
    if CUDA_GPU:
        test_backends.insert(1, ('mylearn.backends._cudamat', 'Cudamat'))
    # contiguous slice testing
    for a_dims, slices, axes in [((5, 5), slice(0, 2), 0),
                                 ((5, 5), slice(0, 2), 1),
                                 ((100, 500), slice(95, 400), 0),
                                 ((100, 500), slice(95, 400), 1)]:
        for backend, classname in test_backends:
            sys.stdout.write("%s\t%dx%d %d:%d %s slice\t%d loop, best of %d:" %
                             (classname, a_dims[0], a_dims[1], slices.start,
                              slices.stop, "row" if axes == 0 else "col",
                              number, repeat))
            sys.stdout.flush()
            sys.stdout.write("\t%f sec\n" % bench_mat_slicing(backend,
                             classname, a_dims, slices, axes, number, repeat))
    # arbitrary index testing
    for a_dims, indices in [((5, 5), [0, 4, 2]),
                            ((100, 500), [50, 10, 90, 22, 95, 0, 5, 9, 95]),
                            ((16000, 16000), [1500, 200, 300, 1599])]:
        for backend, classname in test_backends:
            for name, lop, rop in [('fancy', '[', ']'),
                                   ('take', '.take(', ', 0)')]:
                sys.stdout.write("%s\t%dx%d %d %s indices\t%d loop, "
                                 "best of %d:" % (classname, a_dims[0],
                                                  a_dims[1], len(indices),
                                                  name, number, repeat))
                sys.stdout.flush()
                sys.stdout.write("\t%f sec\n" % bench_mat_indexing(backend,
                                 classname, a_dims, indices, lop, rop, number,
                                 repeat))
