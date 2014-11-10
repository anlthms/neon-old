#!/usr/bin/env/python

from neon.backends.cpu import CPU, CPUTensor
from neon.util.testing import assert_tensor_equal


class TestCPU(object):

    def __init__(self):
        # this code gets called prior to each test
        pass

    def test_empty_creation(self):
        be = CPU()
        tns = be.empty((4, 3))
        assert tns.shape == (4, 3)

    def test_array_creation(self):
        be = CPU()
        tns = be.array([[1, 2], [3, 4]])
        assert tns.shape == (2, 2)
        assert_tensor_equal(tns, CPUTensor([[1, 2], [3, 4]]))
