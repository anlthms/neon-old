#cython: profile=False

import numpy as np
cimport cython
cimport numpy as np

# TODO: create lookup table structs for max_int and max_int - 1

# though we don't use the Numpy C-API, this call prevents compiler warnings
np.import_array()

# define the storage used for each fixed-point element
ctypedef np.int32_t elemtype_t

# overflow handling mechanisms
ctypedef enum ofl_t:
    OFL_SATURATE
    OFL_WRAP

# rounding mechanisms
ctypedef enum rnd_t:
    RND_TRUNCATE
    RND_NEAREST

cpdef inline elemtype_t fp_rescale(elemtype_t inval, int in_sign_bit,
                                   int in_int_bits, int in_frac_bits,
                                   ofl_t overflow, rnd_t rounding,
                                   int out_sign_bit, int out_int_bits,
                                   int out_frac_bits):
    cdef elemtype_t max_int, outval
    outval = inval
    # scale to expected output format
    if in_frac_bits != out_frac_bits:
        if in_frac_bits > out_frac_bits & rounding == RND_TRUNCATE:
            outval = inval >> (in_frac_bits - out_frac_bits)
        elif in_frac_bits < out_frac_bits & rounding == RND_TRUNCATE:
            outval = inval << (out_frac_bits - in_frac_bits)
        else:
            print("unsupported rounding format")
    # handle overflow
    max_int = 1 << (out_int_bits + out_frac_bits + (1 - out_sign_bit))
    if outval >= max_int:
        outval = max_int - 1
    return outval

cpdef inline elemtype_t fixed_from_float(double floatval, int sign_bit,
                                         int int_bits, int frac_bits,
                                         ofl_t overflow, rnd_t rounding):
    cdef elemtype_t fixedval, max_int
    assert frac_bits <= 15
    # truncate / round result
    if rounding == RND_TRUNCATE:
        # truncation done with cast to int
        fixedval = <elemtype_t> (floatval * 2**frac_bits)
    else:
        # assume RND_NEAREST 
        fixedval = <elemtype_t> (floatval * 2**frac_bits + 0.5)
    # perform overflow handling
    max_int = 1 << (int_bits + frac_bits)
    if fixedval < max_int:
        if fixedval < -max_int:
            # negative overflow
            if overflow == OFL_SATURATE:
                fixedval = -max_int
            else:
                # assume OFL_WRAP - undefined for signed ints
                if sign_bit:
                    print("undefined signed neg. overflow wrapping")
                    fixedval = 0
                else:
                    fixedval = fixedval & (max_int - 1)
    elif fixedval >= max_int:
        # positive overflow
        if overflow == OFL_SATURATE:
            fixedval = max_int - 1
        else:
            # assume OFL_WRAP
            fixedval = fixedval & (max_int - 1)
    else:
        # non representable value like nan, inf
        print("non-representable value")
        fixed_val = 0
    return fixedval
    
cpdef inline double fixed_to_float(elemtype_t fixedval, int sign_bit,
                                   int int_bits, int frac_bits,
                                   ofl_t overflow, rnd_t rounding):
    cdef double floatval = <double> fixedval / 2**frac_bits
    return floatval
    
cpdef char* fixed_repr(double floatval, int sign_bit, int int_bits,
                       int frac_bits, ofl_t overflow, rnd_t rounding):
    res = "raw float: %f\nfixed decimal: %+.*f" % (floatval, frac_bits,
           fixed_to_float(fixed_from_float(floatval, sign_bit, int_bits,
                                           frac_bits, overflow, rounding),
                          sign_bit, int_bits, frac_bits, overflow, rounding))
    return res
    
@cython.boundscheck(False)
@cython.wraparound(False)
def naive_dot(np.ndarray[elemtype_t, ndim=2, mode="c"] A not None,
              np.ndarray[elemtype_t, ndim=2, mode="fortran"] B not None,
              np.ndarray[elemtype_t, ndim=2, mode="c"] out not None,
              int sign_bit, int int_bits, int frac_bits, ofl_t overflow,
              rnd_t rounding):
    """
    Performs efficient matrix-matrix multiplication on existing fixed point
    matrices, writing results into a pre-allocated matrix of the same type.

    Assumes that each input matrix has exactly two dimensions, each element of
    each matrix has the same type, the dimensions are amenable for the
    multiply, and that the RHS operand has memory laid out in contiguous
    column order (typical is contiguous row).  This can be done
    without copying by passing in the .T view of the numpy array.
    """
    cdef Py_ssize_t x, y, i
    assert A.shape[1] == B.shape[0]
    assert A.shape[0] == out.shape[0]
    assert B.shape[1] == out.shape[1]
    for x in xrange(out.shape[0]):
        for y in xrange(out.shape[1]):
            out[x, y] = 0
            for i in xrange(A.shape[1]):
                out[x, y] += fp_rescale(A[x, i] * B[i, y], sign_bit,
                                        2 * int_bits, 2 * frac_bits, overflow,
                                        rounding, sign_bit, int_bits, frac_bits)
            # note that we're only rescaling the additions once after
            # accumulating an entire row.  Technically we should rescale after
            # each individual addition but this speeds things up and shouldn't
            # matter if we're doing saturation.
            out[x, y] = fp_rescale(out[x, y], sign_bit, int_bits, frac_bits,
                                   overflow, rounding, sign_bit, int_bits,
                                   frac_bits)
