#!/usr/bin/env/python

from nose.plugins.attrib import attr
from nose.tools import nottest

from neon.util.compat import CUDA_GPU
from neon.util.testing import assert_tensor_equal

if CUDA_GPU:
    # TODO: resolve.  Currently conflicts with cuda RBM tests (can only
    # instantiate once during make test run)
    # from neon.backends.gpu import GPU, GPUTensor
    # be = GPU(rng_seed=0)
    be = None
    GPUTensor = None
    pass


class TestGPU(object):

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_empty_creation(self):
        tns = be.empty((4, 3))
        assert tns.shape == (4, 3)

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_array_creation(self):
        tns = be.array([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)
        assert_tensor_equal(tns, GPUTensor([[1, 2], [3, 4]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_argmin_noaxis(self):
        be = GPU()
        tsr = be.array([[-1, 0], [1, 92]])
        out = be.empty([1, 1])
        be.argmin(tsr, None, out)
        assert_tensor_equal(out, GPUTensor([[0]]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_argmin_axis0(self):
        be = GPU()
        tsr = be.array([[-1, 0], [1, 92]])
        out = be.empty((2, ))
        be.argmin(tsr, 0, out)
        assert_tensor_equal(out, GPUTensor([0, 0]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_argmin_axis1(self):
        be = GPU()
        tsr = be.array([[-1, 10], [11, 9]])
        out = be.empty((2, ))
        be.argmin(tsr, 1, out)
        assert_tensor_equal(out, GPUTensor([0, 1]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_argmax_noaxis(self):
        be = GPU()
        tsr = be.array([[-1, 0], [1, 92]])
        out = be.empty([1, 1])
        be.argmax(tsr, None, out)
        assert_tensor_equal(out, GPUTensor(3))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_argmax_axis0(self):
        be = GPU()
        tsr = be.array([[-1, 0], [1, 92]])
        out = be.empty((2, ))
        be.argmax(tsr, 0, out)
        assert_tensor_equal(out, GPUTensor([1, 1]))

    @attr('cuda')
    @nottest  # TODO: fix based on above
    def test_argmax_axis1(self):
        be = GPU()
        tsr = be.array([[-1, 10], [11, 9]])
        out = be.empty((2, ))
        be.argmax(tsr, 1, out)
        assert_tensor_equal(out, GPUTensor([1, 0]))
