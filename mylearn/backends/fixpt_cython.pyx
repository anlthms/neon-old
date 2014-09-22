#cython: profile=False

import numpy as np
cimport cython
cimport numpy as np

# though we don't use the Numpy C-API, this call prevents compiler warnings
np.import_array()

# define the storage used for each fixed-point element
ctypedef np.int64_t elemtype_t
elemtype = np.int64
ctypedef float elemfloat_t
elemfloat = np.float32

# overflow handling mechanisms
ctypedef enum ofl_t:
    OFL_SATURATE
    OFL_WRAP

# rounding mechanisms
ctypedef enum rnd_t:
    RND_TRUNCATE
    RND_NEAREST

# encapsulation of fixpt type parameters
ctypedef struct fixpt:
    int sign_bit
    int int_bits
    int frac_bits
    ofl_t overflow
    rnd_t rounding

cpdef inline fixpt fixpt_dtype(int sign_bit, int int_bits, int frac_bits,
                               ofl_t overflow, rnd_t rounding):
    cdef fixpt res
    res.sign_bit = sign_bit
    res.int_bits = int_bits
    res.frac_bits = frac_bits
    res.overflow = overflow
    res.rounding = rounding
    return res

cpdef inline elemtype_t fp_rescale(elemtype_t inval, fixpt in_type,
                                   fixpt out_type):
    cdef elemtype_t max_int, outval
    outval = inval
    # scale to expected output format
    if in_type.frac_bits != out_type.frac_bits:
        if ((in_type.frac_bits > out_type.frac_bits) &
            (out_type.rounding == RND_TRUNCATE)):
            outval = inval >> (in_type.frac_bits - out_type.frac_bits)
        elif ((in_type.frac_bits > out_type.frac_bits) & 
              (out_type.rounding == RND_NEAREST)):
            # add 0.5 prior to rescale to nearest
            outval = ((inval + (1 << (in_type.frac_bits -
                                      out_type.frac_bits - 1))) >>
                     (in_type.frac_bits - out_type.frac_bits))
        elif in_type.frac_bits < out_type.frac_bits:
            outval = inval << (out_type.frac_bits - in_type.frac_bits)
        else:
            print("unsupported rounding format")
    # handle overflow
    max_int = 1 << (out_type.int_bits + out_type.frac_bits +
                    (1 - out_type.sign_bit))
    if outval >= max_int:
        outval = max_int - 1
    return outval

cpdef inline elemtype_t fixed_from_float(elemfloat_t floatval, fixpt dtype):
    cdef elemtype_t fixedval, max_int
    assert (dtype.sign_bit + dtype.int_bits + dtype.frac_bits) <= 32
    # truncate / round result
    if dtype.rounding == RND_TRUNCATE:
        # truncation done with cast to int
        fixedval = <elemtype_t> (floatval * 2**dtype.frac_bits)
    else:
        # assume RND_NEAREST 
        fixedval = <elemtype_t> (floatval * 2**dtype.frac_bits + 0.5)
    # perform overflow handling
    max_int = 1 << (dtype.int_bits + dtype.frac_bits)
    if fixedval < max_int:
        if fixedval < -max_int:
            # negative overflow
            if dtype.overflow == OFL_SATURATE:
                fixedval = -max_int
            else:
                # assume OFL_WRAP - undefined for signed ints
                if dtype.sign_bit:
                    print("undefined signed neg. overflow wrapping")
                    fixedval = 0
                else:
                    fixedval = fixedval & (max_int - 1)
    elif fixedval >= max_int:
        # positive overflow
        if dtype.overflow == OFL_SATURATE:
            fixedval = max_int - 1
        else:
            # assume OFL_WRAP
            fixedval = fixedval & (max_int - 1)
    else:
        # non representable value like nan, inf
        print("non-representable value")
        fixed_val = 0
    return fixedval
    
cpdef inline elemfloat_t fixed_to_float(elemtype_t fixedval, fixpt dtype):
    cdef elemfloat_t floatval = <elemfloat_t> fixedval / 2**dtype.frac_bits
    return floatval
    
cpdef char* fixed_repr(elemfloat_t floatval, fixpt dtype):
    res = "raw float: %f\nfixed decimal: %+.*f" % (floatval, dtype.frac_bits,
           fixed_to_float(fixed_from_float(floatval, dtype), dtype))
    return res
    
@cython.boundscheck(False)
@cython.wraparound(False)
def fixed_from_float_array(np.ndarray[elemfloat_t, ndim=2, mode="c"] A not None,
                           fixpt dtype):
    """
    Construct a 2-d fixed-point array from the values in the floating point
    array given.
    """
    cdef Py_ssize_t x, y
    cdef np.ndarray[elemtype_t, ndim=2] res = np.empty_like(A, dtype=elemtype)
    for x in xrange(res.shape[0]):
        for y in xrange(res.shape[1]):
            res[x, y] = fixed_from_float(A[x, y], dtype)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def fixed_to_float_array(np.ndarray[elemtype_t, ndim=2, mode="c"] A not None,
                         fixpt dtype):
    """
    Construct a 2-d floating-point array from the values in the fixed point
    array given.
    """
    cdef Py_ssize_t x, y
    cdef np.ndarray[elemfloat_t, ndim=2] res = np.empty_like(A,
                                                             dtype=elemfloat)
    for x in xrange(res.shape[0]):
        for y in xrange(res.shape[1]):
            res[x, y] = fixed_to_float(A[x, y], dtype)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def fp_rescale_array(np.ndarray[elemtype_t, ndim=2, mode="c"] A not None,
                     fixpt in_dtype, fixpt out_dtype):
    """
    Perform an inplace rescale of the fixed point array passed.
    """
    cdef Py_ssize_t x, y
    for x in xrange(A.shape[0]):
        for y in xrange(A.shape[1]):
            A[x, y] = fp_rescale(A[x, y], in_dtype, out_dtype)
    return A

@cython.boundscheck(False)
@cython.wraparound(False)
def naive_dot(np.ndarray[elemtype_t, ndim=2, mode="c"] A not None,
              np.ndarray[elemtype_t, ndim=2, mode="fortran"] B not None,
              np.ndarray[elemtype_t, ndim=2, mode="c"] out not None,
              fixpt a_dtype, fixpt b_dtype, fixpt out_dtype):
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
    cdef fixpt tmp_dtype
    assert A.shape[1] == B.shape[0]
    assert A.shape[0] == out.shape[0]
    assert B.shape[1] == out.shape[1]
    tmp_dtype.sign_bit = a_dtype.sign_bit | b_dtype.sign_bit
    tmp_dtype.int_bits = a_dtype.int_bits + b_dtype.int_bits
    tmp_dtype.frac_bits = a_dtype.frac_bits + b_dtype.frac_bits
    for x in xrange(out.shape[0]):
        for y in xrange(out.shape[1]):
            out[x, y] = 0
            for i in xrange(A.shape[1]):
                out[x, y] += fp_rescale(A[x, i] * B[i, y], tmp_dtype,
                                        out_dtype)
            # note that we're only rescaling the additions once after
            # accumulating an entire row.  Technically we should rescale after
            # each individual addition but this speeds things up and shouldn't
            # matter if we're doing saturation.  It's also closer to what our
            # hardware does, only scaling after accumulating a block
            out[x, y] = fp_rescale(out[x, y], a_dtype, out_dtype)
