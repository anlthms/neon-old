"""
rbm training and testing class
"""
__author__ = 'arjun'

import pickle
#import rbm_math as rm
import rbm_cudamath as rm
import numpy as np
import ipdb
import time
import gnumpy as gn

class rbm_class:

    def __init__(self, opts):
        #opts is a dict with FILE_LOAD_FLAG=False, numVisible=375, numHidden=100, batchsize=100, maxEpoch=1, eta=0.01, momentum=0.9
        
        #unpack opts for numVisible & numHidden
        numVisible = opts['numVisible']
        numHidden = opts['numHidden']
        
        #initialize the RBM parameters
        if opts['FILE_LOAD_FLAG']:
            self.load_rbm_weights()
        else:
            self.numHidden = numHidden
            self.numVisible = numVisible
            
            self.W = 0.01*gn.randn(numVisible, numHidden)
            self.c = gn.zeros((numVisible, 1))
            self.b = gn.zeros((numHidden, 1))
            #temporary variables for CD
            self.Winc = gn.zeros((numVisible, numHidden))
            self.binc = gn.zeros((numHidden, 1))
            self.cinc = gn.zeros((numVisible, 1))
                    
        #parameters
        self.penalty = .001
        self.eta = opts['eta']
        self.momentum = opts['momentum']
        #avgstart=5
        self.maxEpoch = opts['maxEpoch']
        self.batchsize = opts['batchsize']

    def save_rbm_weights(self, labelsFlag=False):
        if labelsFlag:
            with open('rbm_weights.pkl', 'w') as f:
                pickle.dump([self.W, self.b, self.c, self.numHidden, self.numVisible, self.Wc, self.cc, 
                                self.uniqueClasses, self.numClasses],f)
        else:
            with open('rbm_weights.pkl', 'w') as f:
                pickle.dump([self.W, self.b, self.c, self.numHidden, self.numVisible],f)
    
    def load_rbm_weights(self, labelsFlag=False):
        if labelsFlag:
            with open('rbm_weights.pkl') as f:
                [self.W, self.b, self.c, self.numHidden, self.numVisible, self.Wc, self.cc, 
                                self.uniqueClasses, self.numClasses] = pickle.load(f)            
        else:
            with open('rbm_weights.pkl') as f:
                [self.W, self.b, self.c, self.numHidden, self.numVisible] = pickle.load(f)
            
    #CD-steps
    def applyCD(self, unitActivity):
        #change eta if changing batchsize
        unitActivity = gn.garray(unitActivity)
         
        numTotalSamples = unitActivity.shape[1]
        batchsize=self.batchsize #assuming each training sample is applied independently for CD
        numBatches= int(np.ceil(float(numTotalSamples)/batchsize))
        
        t=1.
        
        for epochs in range(0,self.maxEpoch):
            #if (epochs+1) % 5 == 0: #save the parameters every 5 epochs
            #    self.save_rbm_weights()
            batchsize=self.batchsize
            
            print "Beginning CD epoch #", epochs
            errSum=0
            start_time=time.time()
            for batch in range(0,numBatches):
                #print batch
                if batch == numBatches - 1:
                    batchsize = numTotalSamples - batchsize*(numBatches-1)
                    data = unitActivity[:,(batch*self.batchsize):(batch*self.batchsize)+batchsize ]
                else:
                    data = unitActivity[:,(batch*batchsize):((batch+1)*batchsize) ]
                
                ph = rm.logistic(gn.dot(self.W.T,data) + self.b) 
                nhstates = ph > gn.rand(self.numHidden, batchsize)
                
                #down
                negdata = gn.dot(self.W,nhstates) + self.c 
                negdatastates = rm.logistic(negdata) > gn.rand(negdata.shape[0], negdata.shape[1])

                #up again: non-sparse version
                nh = rm.logistic(gn.dot(self.W.T,negdatastates) + self.b)
                
                dW = gn.dot(data,ph.T) - gn.dot(negdatastates,nh.T) #v0*h0 - v1*h1
                #ipdb.set_trace()
                dc = data - negdatastates
                dc = gn.mean(dc,axis=1)
                db = gn.mean(ph - nh, axis=1)

                self.Winc = self.momentum*self.Winc + self.eta*(dW/batchsize - self.penalty*self.W)
                self.binc = self.momentum*self.binc + self.eta*(db[:,gn.newaxis]/batchsize) #could add penalty here also?
                self.cinc = self.momentum*self.cinc + self.eta*(dc[:,gn.newaxis]/batchsize)

                self.W += self.Winc
                self.b += self.binc
                self.c += self.cinc
                
                l1error = data-negdatastates 
                errSum += abs(l1error).sum() #l1error.multiply(l1error).sum()
            
            print 'Ended epoch ' + str(epochs+1) + '/' + str(self.maxEpoch) +  '; Reconstruction error= ' + str(errSum)
            end_time=time.time()
            print("Elapsed time was %g seconds" % (end_time - start_time))
            
        self.save_rbm_weights()
        return self
    
    def rbmup(self, x):
        #compute the hidden activity given the visibles
        #consider mini-batching if slow
        return rm.logistic(gn.dot(self.W.T,x) + self.b)
    
    
    #CD-steps for last layer with labels
    def applyCDwithLabels(self, unitActivity, labels, resumeFlag=False):
        #change eta if changing batchsize
        labels=gn.garray(labels)
        unitActivity = gn.garray(unitActivity)
        
        numTotalSamples = unitActivity.shape[1]
        batchsize=self.batchsize #assuming each training sample is applied independently for CD
        numBatches= int(np.ceil(float(numTotalSamples)/batchsize))
                
        uniqueClasses = np.unique(labels)
        numClasses = len(uniqueClasses) #0-9 for mnist, assuming this is the format for labels for now
        self.uniqueClasses = uniqueClasses
        self.numClasses = numClasses
        classes = gn.zeros((numClasses ,numTotalSamples))
        for i in range(numTotalSamples):
            #ipdb.set_trace()
            classes[int(labels[i]), i] = 1.
        
        #temporary variables for CD
        if not resumeFlag:
            self.Wcinc = gn.zeros((numClasses, self.numHidden))
            self.ccinc = gn.zeros((numClasses, 1))
            self.Wc = 0.01*gn.randn(numClasses, self.numHidden)
            self.cc = gn.zeros((numClasses, 1))
        
        t=1.

        for epochs in range(0,self.maxEpoch):
            #if (epochs+1) % 5 == 0: #save the parameters every 5 epochs
            #    self.save_rbm_weights()
            batchsize=self.batchsize

            print "Beginning CD epoch #", epochs
            errSum=0
            start_time=time.time()
            for batch in range(0,numBatches):
                #print batch
                if batch == numBatches - 1:
                    batchsize = numTotalSamples - batchsize*(numBatches-1)
                    data = unitActivity[:,(batch*self.batchsize):(batch*self.batchsize)+batchsize ]
                    target = classes[:,(batch*self.batchsize):(batch*self.batchsize)+batchsize ]
                else:
                    data = unitActivity[:,(batch*batchsize):((batch+1)*batchsize) ]
                    target = classes[:,(batch*batchsize):((batch+1)*batchsize) ]

                ph = rm.logistic(gn.dot(self.W.T,data) + gn.dot(self.Wc.T,target) + self.b) 
                nhstates = ph > gn.rand(self.numHidden, batchsize)

                #down
                negdata = gn.dot(self.W,nhstates) + self.c 
                negdatastates = rm.logistic(negdata) > gn.rand()
                negclasses = rm.softmax(gn.dot(self.Wc, nhstates) + self.cc )
                negclassesstates = rm.softmax_sample(negclasses)
                
                #up again: non-sparse version
                nh = rm.logistic(gn.dot(self.W.T,negdatastates) + gn.dot(self.Wc.T,negclassesstates) + self.b)
                
                dW = gn.dot(data,ph.T) - gn.dot(negdatastates,nh.T) #v0*h0 - v1*h1
                dc = data - negdatastates
                dc = gn.mean(dc, axis=1)
                db = gn.mean(ph - nh, axis=1)
                
                dWc = gn.dot(target,ph.T) - gn.dot(negclassesstates,nh.T) #v0*h0 - v1*h1
                dcc = (target - negclassesstates).mean(axis=1)
                
                self.Winc = self.momentum*self.Winc + self.eta*(dW/batchsize - self.penalty*self.W)
                self.binc = self.momentum*self.binc + self.eta*(db[:,gn.newaxis]/batchsize) #could add penalty here also?
                self.cinc = self.momentum*self.cinc + self.eta*(dc[:,gn.newaxis]/batchsize)
                
                self.Wcinc = self.momentum*self.Wcinc + self.eta*(dWc/batchsize - self.penalty*self.Wc)
                self.ccinc = self.momentum*self.ccinc + self.eta*(dcc[:,gn.newaxis]/batchsize)
                
                self.W += self.Winc
                self.b += self.binc
                self.c += self.cinc
                
                self.Wc += self.Wcinc
                self.cc += self.ccinc
                
                l1error = target-negclassesstates 
                errSum += abs(l1error).sum() #l1error.multiply(l1error).sum()
            #ipdb.set_trace()
            print 'Ended epoch ' + str(epochs+1) + '/' + str(self.maxEpoch) +  '; Reconstruction error= ' + str(errSum/2.)
            end_time=time.time()
            print("Elapsed time was %g seconds" % (end_time - start_time))

        self.save_rbm_weights(labelsFlag=True)
        return self
                
    
    #prediction step
    def predict(self, unitActivity):
        #given search query, predict probability of target feature
        #assuming target_feature is boolean for now (e.g. click_bool or booking_bool)

        #method used to predict the target_field
        #PREDICTION_METHOD = 1; free energy method, target_feature represented as single unit
        #PREDICTION_METHOD = 2; probability method, target_feature represented as single unit
        #PREDICTION_METHOD = 3; probability method, target_feature represented as softmax
        #PREDICTION_METHOD = 4; free-energy method, target_feature represented as softmax
        PREDICTION_METHOD = 4

        print "**********************************Generating predictions**********************************"
        start_time=time.time()
        
        numTotalSamples = unitActivity.shape[1]
        batchsize = self.batchsize
        numBatches= int(np.ceil(float(numTotalSamples)/batchsize))

        labels = np.zeros((1, numTotalSamples))
        errSum = 0.

        for batch in range(0,numBatches):
            if (batch + 1) % 10000 == 0:
                print "Batch #", batch+1
            if batch == numBatches - 1:
                batchsize = numTotalSamples - batchsize*(numBatches-1)
                data = unitActivity[:,(batch*self.batchsize):(batch*self.batchsize)+batchsize ]
            else:
                data = unitActivity[:,(batch*batchsize):((batch+1)*batchsize) ]

            
            #probability method, target_feature represented as softmax (2 unit, pool)
            #up
            ph = rm.logistic(gn.dot(self.W.T,data) + self.b) #for CSC matrix change dot to *
            nhstates = ph > gn.rand(self.numHidden, batchsize)

            #down
            #ipdb.set_trace()
            negdata = gn.dot(self.Wc,ph) + self.cc #for COO matrix, will want custom kernel 
            negdata = rm.softmax(negdata)
            #negdatastates = rm.softmax_sample(negdata)[1]
            #negdatastates = negdatastates[1]

            if batch == numBatches - 1:
                labels[:,(batch*self.batchsize):(batch*self.batchsize)+batchsize] = np.argmax(negdata.as_numpy_array(), axis=0)
            else:
                labels[:,(batch*self.batchsize):((batch+1)*self.batchsize)] = np.argmax(negdata.as_numpy_array(), axis=0)

        end_time=time.time()
        print("Finished prediction step. Elapsed time was %g seconds" % (end_time - start_time))

        return labels




