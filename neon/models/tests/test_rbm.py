# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Tests for restricted boltzmann machine (RBM)

- create fake inputs (cudanet class with one small minibatch of 2D data)
- create a fake instance of the RBM class with the model structure from
  yaml replaced by some small weight init / nodes parameters
- precompute the output values we expect for a gradient update and
  numerically compare that we get them.

"""
from nose.plugins.attrib import attr
from nose.tools import nottest
import numpy as np

from neon.models.layer import RBMLayer
from neon.models.learning_rule import GradientDescent
from neon.transforms.logistic import Logistic
from neon.transforms.sum_squared import SumSquaredDiffs
from neon.util.testing import assert_tensor_near_equal
from neon.util.compat import CUDA_GPU


@attr('cuda')
class TestCudaRBM:

    def setup(self):
        if CUDA_GPU:
            from neon.backends.gpu import GPU, GPUTensor

        # TODO: remove randomness from expected target results
        self.be = GPU(rng_seed=0)

        # reusable fake data
        self.inputs = GPUTensor(np.ones((2, 100)))

        # create fake layer
        nin = 2
        conf = {'name': 'testlayer', 'num_nodes': 2,
                'weight_init': {'type': 'normal', 'loc': 0.0, 'scale': 0.01}}
        lr_params = {'learning_rate': 0.01, 'backend': self.be}
        thislr = GradientDescent(name='vis2hidlr', lr_params=lr_params)
        activation = Logistic()
        self.layer = RBMLayer(conf['name'], backend=self.be, batch_size=100,
                              pos=0, learning_rule=thislr,
                              nin=nin, nout=conf['num_nodes'],
                              activation=activation,
                              weight_init=conf['weight_init'])
        # create fake cost
        self.cost = SumSquaredDiffs(olayer=self.layer)

    def test_cudanet_positive(self):
        self.layer.positive(self.inputs)
        target = np.array([0.50541031, 0.50804842],
                          dtype=np.float32)
        assert_tensor_near_equal(self.layer.p_hid_plus.raw()[:, 0], target)

    def test_cudanet_negative(self):
        self.layer.positive(self.inputs)
        self.layer.negative(self.inputs)
        target = np.array([0.50274211,  0.50407821],
                          dtype=np.float32)
        assert_tensor_near_equal(self.layer.p_hid_minus.raw()[:, 0], target)

    @nottest  # TODO: remove randomness
    def test_cudanet_cost(self):
        self.layer.positive(self.inputs)
        self.layer.negative(self.inputs)
        thecost = self.cost.apply_function(self.inputs)
        target = 106.588943481
        assert_tensor_near_equal(thecost, target)
