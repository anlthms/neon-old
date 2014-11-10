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
