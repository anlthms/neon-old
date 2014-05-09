"""
helper math functions for RBM
"""

__author__ = 'arjun'

import numpy as np
import ipdb

def logistic(x):
    return 1/(1 + np.exp(-x))

def softmax(x): #todo: could factorize min(x) in Nr and Dr
    if len(x.shape) == 1 or x.shape[1] == 1:
        tmp = np.exp( x - np.min(x))
        denom = np.sum( tmp )
        mu = tmp / denom
    else:
        tmp = np.exp( x - np.min(x, axis=0))
        denom = np.sum( tmp, axis=0 )
        mu = tmp / denom

    return mu

def softmax_sample(probmat):
    oneofn = np.zeros(probmat.shape)
    sample = np.cumsum(probmat, axis=0)
    if len(probmat.shape) == 1:
        sample = sample > np.random.rand()
        iX = np.argmax(sample, axis=0)
        oneofn[iX] = 1
    else:
        sample = sample > np.random.rand(1, probmat.shape[1])
        iX = np.argmax(sample, axis=0)
        for eachDataPt in range(0, probmat.shape[1]):
            oneofn[iX[eachDataPt], eachDataPt] = 1

    return oneofn