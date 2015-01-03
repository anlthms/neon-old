'''shmarray.py

Shared memory array implementation for numpy which delegates all the nasty
stuff to multiprocessing.sharedctypes.

Copyright (c) 2010, David Baddeley
All rights reserved.'''

# Licensced under the BSD liscence ...
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# Neither the name of the <ORGANIZATION> nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# Modifications: Made PEP8 compliant
# ----------------------------------------------------------------------------
import numpy
from multiprocessing import sharedctypes
from numpy import ctypeslib


class Shmarray(numpy.ndarray):

    '''subclass of ndarray with overridden pickling functions which record
    dtype, shape etc... but defer pickling of the underlying data to the
    original data source.

    Doesn't actually handle allocation of the shared memory - this is done in
    create, and zeros, ones, (or create_copy) are the functions which should
    be used for creating a new shared memory array.

    TODO - add argument checking to ensure that the user is passing reasonable
    values.'''
    def __new__(cls, ctypes_array, shape, dtype=float,
                strides=None, offset=0, order=None):

        # some magic (copied from numpy.ctypeslib) to make sure the ctypes
        # array has the array interface
        tp = type(ctypes_array)
        try:
            tp.__array_interface__
        except AttributeError:
            ctypeslib.prep_array(tp)

        obj = numpy.ndarray.__new__(cls, shape, dtype, ctypes_array, offset,
                                    strides, order)

        # keep track of the underlying storage
        # this may not be strictly necessary as the same info should be stored
        # in .base
        obj.ctypes_array = ctypes_array

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.ctypes_array = getattr(obj, 'ctypes_array', None)

    def __reduce_ex__(self, protocol):
        '''delegate pickling of the data to the underlying storage, but keep
        copies of shape, dtype & strides.

        TODO - find how to get at the offset and order parameters and keep
        track of them as well.'''
        return Shmarray, (self.ctypes_array, self.shape, self.dtype,
                          self.strides)  # , self.offset, self.order)

    def __reduce__(self):
        return __reduce_ex__(self, 0)


def create(shape, dtype='d', alignment=32):
    '''Create an uninitialised shared array. Avoid object arrays, as these
    will almost certainly break as the objects themselves won't be stored in
    shared memory, only the pointers'''
    shape = numpy.atleast_1d(shape).astype('i')
    dtype = numpy.dtype(dtype)

    # we're going to use a flat ctypes array
    n = numpy.prod(shape) + alignment
    # The upper bound of size we want to allocate to be certain
    #  that we can take an aligned array of the right size from it.
    n_bytes_big = n * dtype.itemsize
    # The final (= right) size of the array
    n_bytes_right = numpy.prod(shape) * dtype.itemsize

    dt = 'b'

    # We create the big array first
    a = sharedctypes.RawArray(dt, n_bytes_big)

    sa = Shmarray(a, (n_bytes_big,), dt)

    # We pick the first index of the new array that is aligned
    # If the address of the first element is 1 and we want 8-alignment, the
    #  first aligned index of the array is going to be 7 == -1 % 8
    start_index = -sa.ctypes.data % alignment
    # Finally, we take the (aligned) subarray and reshape it.
    sa = sa[start_index:start_index + n_bytes_right].view(dtype).reshape(shape)

    return sa


def zeros(shape, dtype='d'):
    """Create an shared array initialised to zeros. Avoid object arrays, as
    these will almost certainly break as the objects themselves won't be
    stored in shared memory, only the pointers"""

    sa = create(shape, dtype=dtype)

    # contrary to the documentation, sharedctypes.RawArray does NOT always
    # return an array which is initialised to zero - do it ourselves
    # http://code.google.com/p/python-multiprocessing/issues/detail?id=25
    sa[:] = numpy.zeros(1, dtype)
    return sa


def ones(shape, dtype='d'):
    '''Create an shared array initialised to ones. Avoid object arrays, as
    these will almost certainly break as the objects themselves won't be
    stored in shared memory, only the pointers'''
    sa = create(shape, dtype=dtype)

    sa[:] = numpy.ones(1, dtype)
    return sa


def create_copy(a):
    '''create a a shared copy of an array'''
    # create an empty array
    b = create(a.shape, a.dtype)

    # copy contents across
    b[:] = a[:]

    return b
