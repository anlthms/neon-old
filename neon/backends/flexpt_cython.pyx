#cython: profile=False
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np
cimport cython
cimport numpy as np

# though we don't use the Numpy C-API, this call prevents compiler warnings
np.import_array()

# define the storage used for each flexpoint element
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
    RND_NEAREST_BIASED
    RND_NEAREST_UNBIASED

# encapsulation of flexpt type parameters
ctypedef struct flexpt:
    int sign_bit
    int int_bits
    int frac_bits
    int point_shift
    ofl_t overflow
    rnd_t rounding

cpdef inline flexpt flexpt_dtype(int sign_bit, int int_bits, int frac_bits,
                                 ofl_t overflow, rnd_t rounding,
                                 int point_shift = 0):
    cdef flexpt res
    res.sign_bit = sign_bit
    res.int_bits = int_bits
    res.frac_bits = frac_bits
    res.point_shift = point_shift
    res.overflow = overflow
    res.rounding = rounding
    return res

cpdef inline elemtype_t fp_rescale(elemtype_t inval, flexpt in_type,
                                   flexpt out_type):
    cdef elemtype_t max_int, outval
    cdef int in_scale, out_scale
    outval = inval
    in_scale = in_type.frac_bits + in_type.point_shift
    out_scale = out_type.frac_bits + out_type.point_shift
    # scale to expected output format
    if in_scale != out_scale:
        if (in_scale > out_scale):
            if out_type.rounding == RND_TRUNCATE:
                outval = inval >> (in_scale - out_scale)
            elif (out_type.rounding == RND_NEAREST_BIASED or
                  out_type.rounding == RND_NEAREST_UNBIASED):
                # add 0.5 prior to rescale to nearest
                outval = ((inval + (1 << (in_scale - out_scale - 1))) >>
                          (in_scale - out_scale))
                if (out_type.rounding == RND_NEAREST_UNBIASED and inval < 0 and
                    inval & ((1 << in_scale) - 1) == (1 << (in_scale - 1))):
                    # for unbiased midpoint rounding we want to prevent the
                    # carry in bit from being set (i.e. don't add 0.5)
                    outval = inval >> (in_scale - out_scale)
            else:
                print("unsupported rounding format")
        elif in_scale < out_scale:
            outval = inval << (out_scale - in_scale)
    # handle overflow
    max_int = <elemtype_t> 1 << (out_type.int_bits + out_type.frac_bits +
                    (1 - out_type.sign_bit))
    if outval >= max_int:
        outval = max_int - 1
    return outval

cpdef inline elemtype_t flex_from_float(elemfloat_t floatval, flexpt dtype):
    cdef elemtype_t flexval, max_int
    cdef elemfloat_t multiplier = 2.0**(dtype.frac_bits + dtype.point_shift)
    assert (dtype.sign_bit + dtype.int_bits + dtype.frac_bits) <= 32
    # truncate / round result
    if dtype.rounding == RND_TRUNCATE:
        # truncation done with cast to int
        flexval = <elemtype_t> (floatval * multiplier)
    else:
        if floatval >= 0:
            # works for RND_NEAREST_BIASED and RND_NEAREST_UNBIASED
            flexval = <elemtype_t> (floatval * multiplier + 0.5)
        else:
            if dtype.rounding == RND_NEAREST_BIASED:
                flexval = <elemtype_t> (floatval * multiplier - 0.5)
                if int(floatval) - floatval == 0.5:
                    # midpoint value
                    flexval = <elemtype_t> (floatval * multiplier)
            else: # dtype.rounding == RND_NEAREST_UNBIASED
                flexval = <elemtype_t> (floatval * multiplier - 0.5)
    # perform overflow handling
    max_int = <elemtype_t> 1 << (dtype.int_bits + dtype.frac_bits)
    if flexval < max_int:
        if flexval < -max_int:
            # negative overflow
            if dtype.overflow == OFL_SATURATE:
                flexval = -max_int
            else:
                # assume OFL_WRAP - undefined for signed ints
                if dtype.sign_bit:
                    print("undefined signed neg. overflow wrapping")
                    flexval = 0
                else:
                    flexval = flexval & (max_int - 1)
    elif flexval >= max_int:
        # positive overflow
        if dtype.overflow == OFL_SATURATE:
            flexval = max_int - 1
        else:
            # assume OFL_WRAP
            flexval = flexval & (max_int - 1)
    else:
        # non representable value like nan, inf
        print("non-representable value")
        flexval = 0
    return flexval
    
cpdef inline elemfloat_t flex_to_float(elemtype_t flexval, flexpt dtype):
    cdef elemfloat_t floatval = <elemfloat_t> flexval / (2.0 **
                                                         (dtype.frac_bits +
                                                          dtype.point_shift))
    return floatval
    
@cython.boundscheck(False)
@cython.wraparound(False)
def flex_from_float_array(np.ndarray[elemfloat_t, ndim=2, mode="c"] A not None,
                          flexpt dtype):
    """
    Construct a 2-d flexpoint array from the values in the floating point
    array given.
    """
    cdef Py_ssize_t x, y
    cdef np.ndarray[elemtype_t, ndim=2] res = np.empty_like(A, dtype=elemtype)
    for x in xrange(res.shape[0]):
        for y in xrange(res.shape[1]):
            res[x, y] = flex_from_float(A[x, y], dtype)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def flex_to_float_array(np.ndarray[elemtype_t, ndim=2, mode="c"] A not None,
                        flexpt dtype):
    """
    Construct a 2-d floating-point array from the values in the flexpoint
    array given.
    """
    cdef Py_ssize_t x, y
    cdef np.ndarray[elemfloat_t, ndim=2] res = np.empty_like(A,
                                                             dtype=elemfloat)
    for x in xrange(res.shape[0]):
        for y in xrange(res.shape[1]):
            res[x, y] = flex_to_float(A[x, y], dtype)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def fp_rescale_array(np.ndarray[elemtype_t, ndim=2, mode="c"] A not None,
                     flexpt in_dtype, flexpt out_dtype):
    """
    Perform an inplace rescale of the flexpoint array passed.
    """
    cdef Py_ssize_t x, y
    if (in_dtype.int_bits == out_dtype.int_bits and 
        in_dtype.frac_bits == out_dtype.frac_bits and
        in_dtype.sign_bit == out_dtype.sign_bit):
        # already in the correct scale, short-circuit operation
        return A
    for x in xrange(A.shape[0]):
        for y in xrange(A.shape[1]):
            A[x, y] = fp_rescale(A[x, y], in_dtype, out_dtype)
    return A

@cython.boundscheck(False)
@cython.wraparound(False)
def naive_dot(np.ndarray[elemtype_t, ndim=2, mode="c"] A not None,
              np.ndarray[elemtype_t, ndim=2, mode="fortran"] B not None,
              np.ndarray[elemtype_t, ndim=2, mode="c"] out not None,
              flexpt a_dtype, flexpt b_dtype, flexpt out_dtype):
    """
    Performs naive matrix-matrix multiplication on existing fixed point
    matrices, writing results into a pre-allocated matrix of the same type.

    Assumes that each input matrix has exactly two dimensions, each element of
    each matrix has the same type, the dimensions are amenable for the
    multiply, and that the RHS operand has memory laid out in contiguous
    column order (typical is contiguous row).  This can be done
    without copying by passing in the .T view of the numpy array.
    """
    cdef Py_ssize_t x, y, i
    cdef flexpt tmp_dtype
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
            # accumulating an entire row.  This likely differs from our
            # hardware which rescales after accumulating each row of a 32x32
            # block.
            out[x, y] = fp_rescale(out[x, y], out_dtype, out_dtype)