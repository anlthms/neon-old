"""
Tests for restricted boltzmann machine (RBM)

- create fake inputs (cudanet class with one small minibatch of 2D data)
- create a fake instance of the RBM class with the model structure from
  yaml replaced by some small weight init / nodes parameters
- precompute the output values we expect for a gradient update and
  numerically compare that we get them.

"""
from nose.plugins.attrib import attr
import numpy as np
from neon.models.learning_rule import GradientDescent
from neon.models.layer import RBMLayer
from neon.transforms.logistic import Logistic
from neon.transforms.sum_squared import SumSquaredDiffs
from neon.util.testing import assert_tensor_near_equal
from neon.util.compat import CUDA_GPU

if CUDA_GPU:
    from neon.backends.gpu import GPU, GPUTensor
    be = GPU(rng_seed=0)


class TestCudaRBM:

    @attr('cuda')
    def setup(self):
        # reusable fake data
        self.inputs = GPUTensor(np.ones((2, 100)))

        # create simple backend instance

        # create fake layer
        nin = 2
        conf = {'name': 'testlayer', 'num_nodes': 2,
                'weight_init': {'type': 'normal', 'loc': 0.0, 'scale': 0.01}}
        lr_params = {'learning_rate': 0.01, 'backend': be}
        thislr = GradientDescent(name='vis2hidlr', lr_params=lr_params)
        activation = Logistic()
        self.layer = RBMLayer(conf['name'], backend=be, batch_size=100,
                              pos=0, learning_rule=thislr,
                              nin=nin + 1, nout=conf['num_nodes'] + 1,
                              activation=activation,
                              weight_init=conf['weight_init'])

        # create fake cost
        self.cost = SumSquaredDiffs()

    @attr('cuda')
    def test_cudanet_positive(self):
        self.layer.positive(self.inputs)
        target = [0.50282145,  0.50257355,  0.5021565]
        assert_tensor_near_equal(self.layer.p_hid_plus.raw()[:, 0], target)

    @attr('cuda')
    def test_cudanet_negative(self):
        self.layer.positive(self.inputs)
        self.layer.negative(self.inputs)
        target = [0.5003587,  0.50139761,  0.49878648]
        assert_tensor_near_equal(self.layer.p_hid_minus.raw()[:, 0], target)

    @attr('cuda')
    def test_cudanet_cost(self):
        self.layer.positive(self.inputs)
        self.layer.negative(self.inputs)
        temp = [be.zeros(self.inputs.shape)]
        thecost = self.cost.apply_function(be, self.inputs,
                                           self.layer.x_minus.take(range(
                                               self.layer.x_minus.shape[0] -
                                               1), axis=0), temp)
        target = 106.588943481
        assert_tensor_near_equal(thecost, target)
