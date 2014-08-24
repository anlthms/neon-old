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
from mylearn.util.factory import Factory
from mylearn.backends._cudamat import Cudamat, CudamatTensor

from mylearn.util.testing import assert_tensor_near_equal


# reusable fake data
inputs = CudamatTensor(np.ones((100, 2)))


# create simple backend instance
kwargs = {'rng_seed': 0}
myBackend = Cudamat(**kwargs)  # gives a backend!

# create fake layer
nin = 2
conf = {'name': 'testlayer', 'num_nodes': 2,
        'activation': 'mylearn.transforms.logistic.Logistic',
        'weight_init': {'type': 'normal', 'loc': 0.0, 'scale': 0.01}}
activation = Factory.create(type=conf['activation'])
layer = RBMLayer(conf['name'], myBackend, 100, 0, 0.01, nin + 1,
                 nout=conf['num_nodes'] + 1,
                 activation=activation, weight_init=conf['weight_init'])

# create fake cost
cost = Factory.create(type='mylearn.transforms.sum_squared.SumSquaredDiffs')


@attr('cuda')
def test_cudamat_positive():
    layer.positive(inputs)
    target = [0.50785673,  0.50782728,  0.50173879]
    assert_tensor_near_equal(layer.p_hid_plus.raw()[0], target)


@attr('cuda')
def test_cudamat_negative():
    layer.negative(inputs)
    target = [0.5039286,  0.50391388,  0.50086939]
    assert_tensor_near_equal(layer.p_hid_minus.raw()[0], target)


@attr('cuda')
def test_cudamat_cost():
    # import ipdb; ipdb.set_trace()
    temp = myBackend.zeros(inputs.shape)
    thecost = cost.apply_function(myBackend, inputs, layer.x_minus.take(
                                  range(layer.x_minus.shape[1] - 1), axis=1),
                                  temp)
    target = 24.5629310
    assert_tensor_near_equal(thecost, target, )
