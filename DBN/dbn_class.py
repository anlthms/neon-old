"""
DBN training and testing class
"""
__author__ = 'arjun'

import pickle
#import rbm_class as rc
#import rbm_cudaclass as rc
import rbm_cudamatclass as rc #using cudamat directly
import numpy as np
import ipdb
import time

class dbn_class:

    def __init__(self, opts):
        
        self.sizes = opts['sizes'] 
        self.numFeatures = opts['numFeatures'] 
        
        self.sizes.insert(0, self.numFeatures)
        self.numLayers = len(self.sizes)-1
        self.rbm = dict()
        opts['FILE_LOAD_FLAG'] = False
        
        for eachLayer in range(self.numLayers): 
            opts['numVisible'] = self.sizes[eachLayer]
            opts['numHidden'] = self.sizes[eachLayer+1]
            self.rbm[eachLayer] = rc.rbm_class(opts)
            
    
    def train(self, x, labels=0, USE_ASSOCIATIVE_MEMORY_LABELS=True, resumeFlag=False):
        
        numRBMs = len(self.rbm)
        
        if numRBMs > 1:
            print "Training Layer input -> hidden 1 weights"
            self.rbm[0] = self.rbm[0].applyCD(x)
            for eachRBM in range(1,numRBMs-1):
                print "Training Layer hidden ", eachRBM , " -> hidden ", eachRBM +1
                x = self.rbm[eachRBM-1].rbmup( x)
                self.rbm[eachRBM] = self.rbm[eachRBM].applyCD(x)
            x = self.rbm[numRBMs-2].rbmup( x)
            
        #apply CD with labels for numRBMs-1
        if USE_ASSOCIATIVE_MEMORY_LABELS:
            print "Training weights to top-most layer"
            self.rbm[numRBMs-1] = self.rbm[numRBMs-1].applyCDwithLabels(x,labels, resumeFlag)
        else: #using DBN to pretrain for a NN
            self.rbm[numRBMs-1] = self.rbm[numRBMs-1].applyCD(x)
        
        return self
        
    def test(self, x):
        #predict labels using the DBN
        
        numRBMs = len(self.rbm)
        if numRBMs > 1:
            for eachRBM in range(0,numRBMs-1):
                x = self.rbm[eachRBM].rbmup( x)
        
        return self.rbm[numRBMs-1].predict(x)
        
        
        