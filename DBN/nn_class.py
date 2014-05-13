"""
neural network training and testing class
"""
__author__ = 'arjun'

import pickle
import numpy as np
import ipdb
import time
import copy
import cudamat as cm

class nn_class:

    def __init__(self, opts, dbn=0):
        #initialize the NN parameters 
        #required parameters in opts: numFeatures, sizes, numClasses, FILE_LOAD_FLAG, batchsize, maxEpoch, FROM_DBN_FLAG
        
        cm.cublas_init()
        cm.CUDAMatrix.init_random(1)
        
        self.FROM_DBN_FLAG = opts['FROM_DBN_FLAG']
        self.numFeatures = opts['numFeatures'] 
        self.FILE_LOAD_FLAG = opts['FILE_LOAD_FLAG'] #False
        self.batchsize = opts['batchsize']
        self.maxEpoch = opts['maxEpoch']
        self.out = 'sigmoid' #'linear', or 'sigmoid'
        self.eta = opts['eta']
        
        if self.FROM_DBN_FLAG:
            #initialize the NN with RBM weights
            self.sizes = dbn.sizes
            self.numClasses = opts['numClasses']
            self.sizes.append(self.numClasses)
            self.numLayers = len(self.sizes)
            
            self.W = dict()
            self.b = dict()
            for eachLayer in range(self.numLayers-2):
                self.W[eachLayer] = dbn.rbm[eachLayer].W
                self.b[eachLayer] = dbn.rbm[eachLayer].b    #biases to the hidden layer        
            
        else:       
            #unpack opts for numVisible & numHidden
            self.sizes = opts['sizes']
            self.sizes.insert(0, self.numFeatures)
            self.numClasses = opts['numClasses']
            self.sizes.append(self.numClasses)
        
            self.numLayers = len(self.sizes)        
            print 'number of layers = ', self.numLayers
            self.W = dict()
            self.b = dict()
            for eachLayer in range(self.numLayers-2):
                self.W[eachLayer] = (np.random.rand(self.sizes[eachLayer], self.sizes[eachLayer+1]) - 0.5) * 2. * 4. * np.sqrt(6. / (self.sizes[eachLayer] + self.sizes[eachLayer+1] +1)) 
                self.b[eachLayer] = np.zeros((self.sizes[eachLayer+1], 1)) #(np.random.rand(self.sizes[eachLayer+1], 1) - 0.5) * 2. * 4. * np.sqrt(6 / (self.sizes[eachLayer] + self.sizes[eachLayer+1] +1))
                #self.W[eachLayer] = 0.01*np.random.randn( self.sizes[eachLayer], self.sizes[eachLayer+1]) 
                #self.b[eachLayer] = 0.01*np.random.randn( self.sizes[eachLayer+1], 1)

        eachLayer = self.numLayers-2
        if self.out == 'sigmoid':
            self.W[eachLayer] = (np.random.rand(self.sizes[eachLayer], self.sizes[eachLayer+1]) - 0.5) * 2. * 4. * np.sqrt(6. / (self.sizes[eachLayer] + self.sizes[eachLayer+1] +1)) 
            self.b[eachLayer] = np.zeros((self.sizes[eachLayer+1], 1))   #(np.random.rand(self.sizes[eachLayer+1], 1) - 0.5) * 2. * 4. * np.sqrt(6 / (self.sizes[eachLayer] + self.sizes[eachLayer+1] +1))
        elif self.out == 'linear':
            self.W[eachLayer] = 0.01*np.random.randn( self.sizes[eachLayer], self.sizes[eachLayer+1])  #these are the weights from the hidden layer to output layer
            self.b[eachLayer] = np.zeros((self.sizes[eachLayer+1], 1))  
            #self.b[self.numLayers-1] = 0.01*np.random.randn( self.sizes[eachLayer+1], 1)
            
            
        #parameters
        self.penalty = .0001
        self.momentum = 0.9
        #avgstart=5
        self.dropoutFraction = 0 #what fraction to dropout
        self.actfunc = 'sigmoid' #'sigm' 'tanh' 'relu'
                
    def tanh_opt(self, a):
        a.mult(2./3.)
        a.apply_tanh()
        a.mult(1.7159)
        return a 
    
    def to_cudamat(self, labelsFlag=False):
        for eachLayer in range(self.numLayers-1):
            self.W[eachLayer] = cm.CUDAMatrix(self.W[eachLayer])
            self.b[eachLayer] = cm.CUDAMatrix(self.b[eachLayer])
            self.Winc[eachLayer] = cm.CUDAMatrix(self.Winc[eachLayer])
            self.binc[eachLayer] = cm.CUDAMatrix(self.binc[eachLayer])

    def to_numpy(self, labelsFlag=False):
        for eachLayer in range(self.numLayers-1):
            self.W[eachLayer] = self.W[eachLayer].asarray()
            self.b[eachLayer] = self.b[eachLayer].asarray()
            self.Winc[eachLayer] = self.Winc[eachLayer].asarray()
            self.binc[eachLayer] = self.binc[eachLayer].asarray()
    
    def relu(self, x):
        y = cm.empty((x.shape[0], x.shape[1]))
        x.greater_than(0., target=y)
        x.mult(y)
        return x

    def apply_actfunc(self, x):
        if self.actfunc == 'sigmoid':
            return x.apply_sigmoid()
        elif self.actfunc == 'relu':
            return self.relu(x)
        elif self.actfunc =='tanh':
            return self.tanh_opt(x)
        
    #backpropagation steps
    def train(self, features, target):
        #change eta if changing batchsize
        batchsize=self.batchsize 
        numTotalSamples = features.shape[1]
        numBatches= int(np.ceil(float(numTotalSamples)/batchsize))
        
        #temporary variables for weight updates
        self.Winc = dict()
        self.binc = dict()
        r = dict()
        h = dict()
        err = dict()
        tmp = dict()
        
        for eachLayer in range(self.numLayers-1):
            self.Winc[eachLayer] = np.zeros_like(self.W[eachLayer])
            self.binc[eachLayer] = np.zeros_like(self.b[eachLayer])
            r[eachLayer] = cm.empty((self.sizes[eachLayer], self.batchsize)) #dropout mask
            
        for eachLayer in range(self.numLayers):           
            h[eachLayer] = cm.empty((self.sizes[eachLayer], self.batchsize)) #activation
            err[eachLayer] = cm.empty((self.sizes[eachLayer], self.batchsize)) #error for backprop
            tmp[eachLayer] = cm.empty((self.sizes[eachLayer], self.batchsize)) #temporary storage for sigmoid/tanh derivative
            
        t=1.
        
        cur_err = np.inf
        
        self.to_cudamat()
        features = cm.CUDAMatrix(features)
        target = cm.CUDAMatrix(target)
        
        if self.out == 'sigmoid':
            mineta =.0001
        elif self.out =='linear':
            mineta =.001
        # if self.actfunc =='relu':
        #     self.eta = .001
        #     mineta = .0001
        RANK_UPDATES=False
        SHUFFLE_FLAG=False #for gfx card difficult to shuffle minibatches?
        
        
        for epochs in range(self.maxEpoch):
            #if (epochs+1) % 5 == 0: #save the parameters every 5 epochs
            #    self.save_nn_weights()
            batchsize = self.batchsize
            print "Beginning backprop epoch #", epochs
            errSum = 0
            start_time = time.time()
            
            #save the weights in case error gets worse: ignore for now as it requires expensive host copy operation
            #curW = dict()
            #for eachLayer in range(self.numLayers-1):
            #    curW[eachLayer] = copy.deepcopy(self.W[eachLayer])
            #    curb[eachLayer] = copy.deepcopy(self.b[eachLayer])
            
            #pre computed values to help in tanh evaluation
            z1 = 1.7159 * 2./3. 
            z2 = 1./(1.7159)**2
            momentum = self.momentum

            #randomize the indices for the minibatch
            if SHUFFLE_FLAG:
                idx = np.random.permutation(numTotalSamples) #not supported on cuda for now [ investigate]
            else:
                idx = range(numTotalSamples)
                
            for batch in range(numBatches):
                if batch == numBatches - 1: #assuming batchsize is evenly divisible by numTotalSamples for now
                    batchsize = numTotalSamples - batchsize*(numBatches-1)
                    data = features.slice(idx[(batch*self.batchsize)], idx[(batch*self.batchsize)]+batchsize)
                    batchTargetData = target.slice(idx[(batch*self.batchsize)], idx[(batch*self.batchsize)]+batchsize)
                    #batchTargetData = batchTargetData[:,np.newaxis ].transpose()
                else:
                    data = features.slice(idx[(batch*batchsize)], idx[((batch+1)*batchsize)] )
                    batchTargetData = target.slice(idx[(batch*batchsize)], idx[((batch+1)*batchsize)])
                
                h[0].assign(data)
                
                if self.dropoutFraction > 0.: 
                    r[eachLayer].fill_with_rand()
                    #dropout fraction for input layer is .2 regardless of dropout for other layers (if >0.)
                    r[eachLayer].greater_than(0.2)
                    h[0].mult(r[eachLayer])
                
                #apply momentum
                for eachLayer in range(self.numLayers-1):
                    self.Winc[eachLayer].mult(self.momentum)
                    self.binc[eachLayer].mult(self.momentum)
                
                #feedforward pass
                for eachLayer in range(self.numLayers-2):
                    cm.dot(self.W[eachLayer].T, h[eachLayer], target=h[eachLayer+1])
                    h[eachLayer+1].add_col_vec(self.b[eachLayer])
                    h[eachLayer+1] = self.apply_actfunc(h[eachLayer+1]) 
                    if self.dropoutFraction > 0.:
                        r[eachLayer+1].fill_with_rand()
                        r[eachLayer+1].greater_than(self.dropoutFraction)
                        h[eachLayer+1].mult(r[eachLayer+1])
                
                eachLayer = self.numLayers - 2
                cm.dot(self.W[eachLayer].T, h[eachLayer], target=h[eachLayer+1])
                h[eachLayer+1].add_col_vec(self.b[eachLayer])
                if self.out =='sigmoid':
                    h[eachLayer+1].apply_sigmoid()
                elif self.out == 'linear':
                    a1 = 1. #dummy , no op
                
                eachLayer = self.numLayers - 1
                h[eachLayer].subtract(batchTargetData, target=err[eachLayer]) #this is -err in previous implementation
                errSum += err[self.numLayers-1].manhattan_norm() 
                
                #compute backpropogating derivatives
                if self.out == 'sigmoid': 
                    err[eachLayer].mult(h[eachLayer])
                    h[eachLayer].mult(-1., target=tmp[eachLayer])
                    tmp[eachLayer].add(1.)
                    err[eachLayer].mult(tmp[eachLayer])
                elif self.out == 'linear':
                    a1 = 1. #dummy, no op
                
                for eachLayer in range(self.numLayers-2, 0, -1):
                    
                    cm.dot(self.W[eachLayer], err[eachLayer+1], target=err[eachLayer])
                    if self.actfunc == 'sigmoid':
                        err[eachLayer].mult(h[eachLayer])
                        h[eachLayer].mult( -1., target=tmp[eachLayer])
                        tmp[eachLayer].add(1.)
                        err[eachLayer].mult(tmp[eachLayer])
                    elif self.actfunc == 'tanh':
                        cm.pow(h[eachLayer],2,target=tmp[eachLayer])
                        tmp[eachLayer].mult(-z2)
                        tmp[eachLayer].add(1.)
                        tmp[eachLayer].mult(z1)
                        err[eachLayer].mult(tmp[eachLayer])
                    elif self.actfunc == 'relu':
                        tmp[eachLayer].assign(h[eachLayer])
                        tmp[eachLayer].greater_than(0.)
                        err[eachLayer].mult(tmp[eachLayer])
                        
                #compute the weight updates
                for eachLayer in range(self.numLayers-1):
                    self.binc[eachLayer].add_sums(err[eachLayer+1], axis=1)
                    self.binc[eachLayer].mult(self.eta/batchsize)
                    if self.dropoutFraction > 0.:
                        h[eachLayer].mult(r[eachLayer])
                    self.Winc[eachLayer].add_dot(h[eachLayer], err[eachLayer+1].T, mult=self.eta/batchsize)
                    self.Winc[eachLayer].add_mult(self.W[eachLayer], alpha=self.penalty*self.eta)
                    
                    #apply the weight updates
                    self.W[eachLayer].subtract(self.Winc[eachLayer])
                    self.b[eachLayer].subtract(self.b[eachLayer])
                                    
                
            #ipdb.set_trace()
            print 'Ended epoch ' + str(epochs+1) + '/' + str(self.maxEpoch) +  '; Reconstruction error= ' + str(errSum)
            if errSum >= cur_err and not RANK_UPDATES:
                self.eta /= 2.
                print 'halving eta to ', self.eta
                #todo: restore weights
                
            else:
                cur_err = errSum
                
            if self.eta< mineta:
                #self.save_nn_weights()
                self.to_numpy()
                return self
            
            end_time=time.time()
            print("Elapsed time was %g seconds" % (end_time - start_time))
        
        self.to_numpy()
        #self.save_nn_weights()
        return self

    #prediction step
    def test(self, features):
                
        print "**********************************Generating predictions**********************************"
        start_time=time.time()
        
        batchsize = self.batchsize
        numTotalSamples = features.shape[1]
        numBatches= int(np.ceil(float(numTotalSamples)/batchsize))

        labels = np.zeros((1, numTotalSamples))
        errSum = 0.
        
        self.to_cudamat()
        features = cm.CUDAMatrix(features)
        idx = cm.empty((1, self.batchsize))
        h = dict()
        for eachLayer in range(self.numLayers):
            h[eachLayer] = cm.empty((self.sizes[eachLayer], self.batchsize)) #activation
        
        for batch in range(0,numBatches):
            if (batch + 1) % 10000 == 0:
                print "Batch #", batch+1
            if batch == numBatches - 1:
                batchsize = numTotalSamples - batchsize*(numBatches-1)
                data = features.slice((batch*self.batchsize), (batch*self.batchsize)+batchsize)
            else:
                data = features.slice((batch*batchsize), ((batch+1)*batchsize) )
            h[0].assign(data)
            
            if self.dropoutFraction > 0.:
                h[0].mult(.8) #to account for dropout of 0.2 on input layer
            
            #feedforward pass
            for eachLayer in range(self.numLayers-2):
                cm.dot(self.W[eachLayer].T, h[eachLayer], target=h[eachLayer+1])
                h[eachLayer+1].add_col_vec(self.b[eachLayer])
                h[eachLayer+1] = self.apply_actfunc(h[eachLayer+1]) 
                if self.dropoutFraction > 0.:
                    h[eachLayer+1].mult(1-self.dropoutFraction)
            
            eachLayer = self.numLayers - 2
            cm.dot(self.W[eachLayer].T, h[eachLayer], target=h[eachLayer+1])
            h[eachLayer+1].add_col_vec(self.b[eachLayer])
            if self.out =='sigmoid':
                h[eachLayer+1].apply_sigmoid()
            elif self.out == 'linear':
                a1 = 1. #dummy , no op

            h[eachLayer+1].argmax(axis=0, target=idx)
            
            if batch == numBatches - 1:
                labels[:,(batch*self.batchsize):(batch*self.batchsize)+batchsize] = idx.asarray()
            else:
                labels[:,(batch*self.batchsize):((batch+1)*self.batchsize)] = idx.asarray()
                        
        end_time=time.time()
        print("Finished prediction step. Elapsed time was %g seconds" % (end_time - start_time))
        
        self.to_numpy()
        return labels

        



