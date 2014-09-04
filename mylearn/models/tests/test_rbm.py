"""
Tests for restricted boltzmann machine (RBM)
After discussing with Scott, we want:
- create fake inputs (cudamat class with one small minibatch of 2D data)
- create a fake instance of the RBM class with the model structure from
  yaml replaced by some small weight init / nodes parameters
- precompute the output values we expect for a gradient update and
  numerically compare that we get them.

"""
from nose.plugins.attrib import attr
import numpy as np

from mylearn.models.layer import RBMLayer
from mylearn.transforms.logistic import Logistic
from mylearn.transforms.sum_squared import SumSquaredDiffs
from mylearn.util.testing import assert_tensor_near_equal
from mylearn.util.compat import CUDA_GPU

if CUDA_GPU:
    from mylearn.backends._cudamat import Cudamat, CudamatTensor


class TestCudaRBM:

    @attr('cuda')
    def setup_class(self):
        # reusable fake data
        self.inputs = CudamatTensor(np.ones((100, 2)))

        # create simple backend instance
        kwargs = {'rng_seed': 0}
        self.myBackend = Cudamat(**kwargs)  # gives a backend!

        # create fake layer
        nin = 2
        conf = {'name': 'testlayer', 'num_nodes': 2,
                'weight_init': {'type': 'normal', 'loc': 0.0, 'scale': 0.01}}
        activation = Logistic()
        self.layer = RBMLayer(conf['name'], self.myBackend, 100, 0, 0.01,
                              nin + 1, nout=conf['num_nodes'] + 1,
                              activation=activation,
                              weight_init=conf['weight_init'])

        # create fake cost
        self.cost = SumSquaredDiffs()

    @attr('cuda')
    def test_cudamat_positive(self):
        self.layer.positive(self.inputs)
        target = [0.50785673,  0.50782728,  0.50173879]
        assert_tensor_near_equal(self.layer.p_hid_plus.raw()[0], target)

    @attr('cuda')
    def test_cudamat_negative(self):
        self.layer.negative(self.inputs)
        target = [0.5039286,  0.50391388,  0.50086939]
        assert_tensor_near_equal(self.layer.p_hid_minus.raw()[0], target)

    @attr('cuda')
    def test_cudamat_cost(self):
        # import ipdb; ipdb.set_trace()
        temp = [self.myBackend.zeros(self.inputs.shape)]
        thecost = self.cost.apply_function(self.myBackend, self.inputs,
                                           self.layer.x_minus.take(range(
                                               self.layer.x_minus.shape[1] -
                                               1), axis=1), temp)
        target = 24.5629310
        assert_tensor_near_equal(thecost, target)
