"""
Tests for restricted boltzmann machine (RBM)
"""

import numpy as np
from layer import RBMLayer

def test_something():
	# things to pass: self.backend, nin, self.layers[i]
    myLayer = RBMLayer(conf['name'], backend, nin + 1,
                     nout=conf['num_nodes']+1,  # (u) bias for both layers
                     activation=activation,
                     weight_init=conf['weight_init'])
    assert True

def test_sigmoid():
    assert True

def test_squared_error():
    assert True

def test_positive():
    assert True

def test_negative():
    assert True




   def fit(self, datasets):
        """
        minimal fit to compare for testing...
        """
        pass
