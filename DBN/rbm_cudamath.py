"""
helper math functions for RBM
"""

__author__ = 'arjun'

import gnumpy as gn
import numpy as np
import ipdb

def logistic(x):
    return 1/(1 + gn.exp(-x))

def softmax(x): #todo: could factorize min(x) in Nr and Dr
    if len(x.shape) == 1 or x.shape[1] == 1:
        tmp = gn.exp( x - gn.min(x))
        denom = gn.sum( tmp )
        mu = gn.garray(tmp / denom)
    else:
        tmp = gn.exp( x - gn.min(x, axis=0))
        denom = gn.sum( tmp, axis=0 )
        mu = gn.garray(tmp / denom)

    return mu

def softmax_sample(probmat):
    oneofn = gn.zeros(probmat.shape)
    sample = np.cumsum(probmat.as_numpy_array(), axis=0) #need cudamat/gnumpy cumsum kernel
    if len(probmat.shape) == 1:
        #sample = sample > gn.rand()
        sample = sample > np.random.rand()
        #index = gn.where(gn.max(sample) == sample)[0] 
        #iX = gn.min(index)
        iX = np.argmax(sample, axis=0)
        oneofn[iX] = 1
    else:
        #sample = sample > gn.rand(1, probmat.shape[1])
        sample = sample > np.random.rand(1, probmat.shape[1])
        iX = np.argmax(sample, axis=0)
        for eachDataPt in range(0, probmat.shape[1]):
            oneofn[iX[eachDataPt], eachDataPt] = 1

    return oneofn
    
    
    # sample = gn.garray(sample > gn.rand())
    # index = gn.where(gn.max(sample) == sample)[0] 
    # index = gn.min(index)