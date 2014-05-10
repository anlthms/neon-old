"""
rbm training and testing class, using cudamat library
"""
__author__ = 'arjun'

import pickle
#import rbm_math as rm
#import rbm_cudamath as rm
import numpy as np
import ipdb
import time
import cudamat as cm

class rbm_class:

    def __init__(self, opts):
        #opts is a dict with FILE_LOAD_FLAG=False, numVisible=375, numHidden=100, batchsize=100, maxEpoch=1, eta=0.01, momentum=0.9
        cm.cublas_init()
        cm.CUDAMatrix.init_random(1)
        
        #unpack opts for numVisible & numHidden
        numVisible = opts['numVisible']
        numHidden = opts['numHidden']
        
        #initialize the RBM parameters
        if opts['FILE_LOAD_FLAG']:
            self.load_rbm_weights()
        else:
            self.numHidden = numHidden
            self.numVisible = numVisible
            
            self.W = cm.CUDAMatrix(0.01*np.random.randn(numVisible, numHidden))
            self.c = cm.CUDAMatrix(np.zeros((numVisible, 1)))
            self.b = cm.CUDAMatrix(np.zeros((numHidden, 1)))
            #temporary variables for CD
            self.Winc = cm.CUDAMatrix(np.zeros((numVisible, numHidden)))
            self.binc = cm.CUDAMatrix(np.zeros((numHidden, 1)))
            self.cinc = cm.CUDAMatrix(np.zeros((numVisible, 1)))
                    
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
                pickle.dump([self.W.numpy_array, self.b.numpy_array, self.c.numpy_array, self.numHidden, 
                                self.numVisible, self.Wc.numpy_array, self.cc.numpy_array, 
                                self.uniqueClasses, self.numClasses],f)
        else:
            with open('rbm_weights.pkl', 'w') as f:
                pickle.dump([self.W.numpy_array, self.b.numpy_array, self.c.numpy_array, self.numHidden, self.numVisible],f)
    
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
        unitActivity = cm.CUDAMatrix(unitActivity)
         
        numTotalSamples = unitActivity.shape[1]
        batchsize=self.batchsize #assuming each training sample is applied independently for CD
        numBatches= int(np.ceil(float(numTotalSamples)/batchsize))
        
        t=1.
        
        #might need to move this in a clever way inside for loop if last batch is different size
        v = cm.empty((self.numVisible, self.batchsize))
        h = cm.empty((self.numHidden, self.batchsize))
        r = cm.empty((self.numHidden, self.batchsize))
        
        for epochs in range(0,self.maxEpoch):
            #if (epochs+1) % 5 == 0: #save the parameters every 5 epochs
            #    self.save_rbm_weights()
            batchsize=self.batchsize
            
            print "Beginning CD epoch #", epochs
            err=[]
            start_time=time.time()
            for batch in range(0,numBatches):
                #print batch
                if batch == numBatches - 1:
                    batchsize = numTotalSamples - batchsize*(numBatches-1)
                    data = unitActivity.slice( (batch*self.batchsize), (batch*self.batchsize)+batchsize ) #might need to do a data.assign
                    v.assign(data)
                else:
                    data = unitActivity.slice( (batch*batchsize), ((batch+1)*batchsize) )
                    v.assign(data)
                
                #apply momentum
                self.Winc.mult(self.momentum)
                self.binc.mult(self.momentum)
                self.cinc.mult(self.momentum)
                
                #ph = rm.logistic(gn.dot(self.W.T,data) + self.b) 
                cm.dot(self.W.T, v, target=h)
                h.add_col_vec(self.b)
                h.apply_sigmoid()
                
                #compute v0*h0
                self.Winc.add_dot(v,h.T) # gn.dot(v,h.T) term in dW
                self.binc.add_sums(h, axis=1)
                self.cinc.add_sums(v, axis=1)
                
                #sample hiddens: nhstates = ph > gn.rand(self.numHidden, batchsize)
                r.fill_with_rand()
                r.less_than(h, target = h)
                
                #down: negdata = gn.dot(self.W,nhstates) + self.c 
                cm.dot(self.W, h, target=v)
                v.add_col_vec(self.c)
                v.apply_sigmoid()
                
                #ignore for now: negdatastates = rm.logistic(negdata) > gn.rand()
                
                #up again: non-sparse version; nh = rm.logistic(gn.dot(self.W.T,negdatastates) + self.b)
                cm.dot(self.W.T, v, target = h)
                h.add_col_vec(self.b)
                h.apply_sigmoid()
                
                #weight updates
                self.Winc.subtract_dot(v, h.T) #gn.dot(negdatastates,nh.T)
                self.cinc.add_sums(v, axis = 1, mult = -1.)
                self.binc.add_sums(h, axis = 1, mult = -1.)
                
                # update weights: todo: add penalty
                self.W.add_mult(self.Winc, self.eta/batchsize) #for penalty : - self.eta*self.penalty*self.W)
                self.c.add_mult(self.cinc, self.eta/batchsize)
                self.b.add_mult(self.binc, self.eta/batchsize)

                # calculate reconstruction error
                v.subtract(data)
                err.append(v.euclid_norm()**2/(self.numVisible*batchsize))
                
            print 'Ended epoch ' + str(epochs+1) + '/' + str(self.maxEpoch) +  '; Reconstruction error= ' + str(np.mean(err))
            end_time=time.time()
            print("Elapsed time was %g seconds" % (end_time - start_time))
            
        self.save_rbm_weights()
        return self

#todo: continue cudamat conversion from here onwards
    
    def rbmup(self, x):
        #compute the hidden activity given the visibles
        #consider mini-batching if slow
        #return rm.logistic(gn.dot(self.W.T,x) + self.b)
        x = cm.CUDAMatrix(x)
        up = cm.CUDAMatrix(np.ones((self.W.shape[1], x.shape[1])))
        cm.dot(self.W.T, x, target=up)
        up.add_col_vec(self.b)
        return up.numpy_array #could have smarter check to keep this on gpu
        
    def softmax(self, x):
        
        if len(x.shape) == 1 or x.shape[1] == 1:
            cm.exp( x.subtract(x.min()), target=self.tmp)
            cm.sum( self.tmp, target=self.denom )
            self.tmp.divide( self.denom, target=self.mu)
        else:
            cm.exp( x.add_row_vec( x.min(axis=0).mult(-1.)), target=self.tmp)
            cm.sum( self.tmp, axis=0 , target=self.denom)
            self.tmp.div_by_row( self.denom, target=self.mu)

        return self.mu
    
    #CD-steps for last layer with labels
    def applyCDwithLabels(self, unitActivity, labels, resumeFlag=False):
        #change eta if changing batchsize
        unitActivity = cm.CUDAMatrix(unitActivity)
        
        numTotalSamples = unitActivity.shape[1]
        batchsize=self.batchsize #assuming each training sample is applied independently for CD
        numBatches= int(np.ceil(float(numTotalSamples)/batchsize))
                
        uniqueClasses = np.unique(labels)
        numClasses = len(uniqueClasses) #0-9 for mnist, assuming this is the format for labels for now
        self.uniqueClasses = uniqueClasses
        self.numClasses = numClasses
        classes = np.zeros((numClasses ,numTotalSamples))
        for i in range(numTotalSamples):
            #ipdb.set_trace()
            classes[int(labels[i]), i] = 1.
        
        classes = cm.CUDAMatrix(classes)
        labels = cm.CUDAMatrix(labels[np.newaxis,:])
        
        #temporary variables for CD
        if not resumeFlag:
            self.Wcinc = cm.CUDAMatrix(np.zeros((numClasses, self.numHidden)))
            self.ccinc = cm.CUDAMatrix(np.zeros((numClasses, 1)))
            self.Wc = cm.CUDAMatrix(0.01*np.random.randn(numClasses, self.numHidden))
            self.cc = cm.CUDAMatrix(np.zeros((numClasses, 1)))
        
        #might need to move this in a clever way inside for loop if last batch is different size
        v = cm.empty((self.numVisible, self.batchsize))
        vt = cm.empty((self.numClasses, self.batchsize))
        h = cm.empty((self.numHidden, self.batchsize))
        ht = cm.empty((self.numHidden, self.batchsize))
        r = cm.empty((self.numHidden, self.batchsize))
        
        t=1.
        
        self.tmp = cm.empty((self.numClasses, self.batchsize))
        self.denom = cm.empty((1, self.batchsize))
        self.mu = cm.empty((self.numClasses, self.batchsize))

        for epochs in range(0,self.maxEpoch):
            #if (epochs+1) % 5 == 0: #save the parameters every 5 epochs
            #    self.save_rbm_weights()
            batchsize=self.batchsize

            print "Beginning CD epoch #", epochs
            err = []
            err2 = []
            start_time=time.time()
            for batch in range(0,numBatches):
                #print batch
                if batch == numBatches - 1:
                    batchsize = numTotalSamples - batchsize*(numBatches-1)
                    data = unitActivity.slice((batch*self.batchsize),(batch*self.batchsize)+batchsize )
                    target = classes.slice( (batch*self.batchsize), (batch*self.batchsize)+batchsize )
                    v.assign(data)
                    vt.assign(target)
                else:
                    data = unitActivity.slice((batch*batchsize), ((batch+1)*batchsize) )
                    target = classes.slice( (batch*batchsize), ((batch+1)*batchsize) )
                    v.assign(data)
                    vt.assign(target)
                
                #apply momentum
                self.Winc.mult(self.momentum)
                self.binc.mult(self.momentum)
                self.cinc.mult(self.momentum)
                self.Wcinc.mult(self.momentum)
                self.ccinc.mult(self.momentum)
                
                #ph = rm.logistic(gn.dot(self.W.T,data) + gn.dot(self.Wc.T,target) + self.b) 
                cm.dot(self.W.T, v, target=h)
                h.add_col_vec(self.b)
                cm.dot(self.Wc.T, vt, target=ht)
                h.add(ht)
                h.apply_sigmoid()
                
                #compute v0*h0
                self.Winc.add_dot(v,h.T) # gn.dot(v,h.T) term in dW
                self.binc.add_sums(h, axis=1)
                self.cinc.add_sums(v, axis=1)
                self.Wcinc.add_dot(vt, h.T) 
                self.ccinc.add_sums(vt, axis=1)
                
                #sample hiddens: nhstates = ph > gn.rand(self.numHidden, batchsize)
                r.fill_with_rand()
                r.less_than(h, target = h)
                
                #down: negdata = gn.dot(self.W,nhstates) + self.c 
                cm.dot(self.W, h, target=v)
                v.add_col_vec(self.c)
                v.apply_sigmoid()
                #ignore for now: negdatastates = rm.logistic(negdata) > gn.rand()
                
                #negclasses = rm.softmax(gn.dot(self.Wc, nhstates) + self.cc )
                #ignore for now: negclassesstates = rm.softmax_sample(negclasses)
                cm.dot(self.Wc, h, target=vt)
                vt.add_col_vec(self.cc)
                vt = self.softmax(vt)
                
                #up again: nh = rm.logistic(gn.dot(self.W.T,negdatastates) + gn.dot(self.Wc.T,negclassesstates) + self.b)
                cm.dot(self.W.T, v, target = h)
                h.add_col_vec(self.b)
                cm.dot(self.Wc.T, vt, target=ht)
                h.add(ht)
                h.apply_sigmoid()
                
                #weight updates
                self.Winc.subtract_dot(v, h.T) #gn.dot(negdatastates,nh.T)
                self.cinc.add_sums(v, axis = 1, mult = -1.)
                self.binc.add_sums(h, axis = 1, mult = -1.)
                self.Wcinc.subtract_dot(vt, h.T) #dWc = gn.dot(target,ph.T) - gn.dot(negclassesstates,nh.T) #v0*h0 - v1*h1
                self.ccinc.add_sums(vt, axis=1, mult = -1.) #dcc = (target - negclassesstates).mean(axis=1)
                
                # update weights: todo: add penalty
                self.W.add_mult(self.Winc, self.eta/batchsize) #for penalty : - self.eta*self.penalty*self.W)
                self.c.add_mult(self.cinc, self.eta/batchsize)
                self.b.add_mult(self.binc, self.eta/batchsize)
                self.Wc.add_mult(self.Wcinc, self.eta/batchsize) #for penalty : - self.eta*self.penalty*self.W)
                self.cc.add_mult(self.ccinc, self.eta/batchsize)
                
                # calculate reconstruction error
                v.subtract(data)
                vt.subtract(target)
                err.append((v.euclid_norm()**2 + vt.euclid_norm()**2)/(self.numVisible*batchsize))
                #err2.append(vt.manhattan_norm()/(self.numVisible*batchsize))
                
            print 'Ended epoch ' + str(epochs+1) + '/' + str(self.maxEpoch) +  '; Reconstruction error= ' + str(np.mean(err))  #+ '; Misclassification rate: ' + str(np.mean(err2))
            end_time=time.time()
            print("Elapsed time was %g seconds" % (end_time - start_time))

        self.save_rbm_weights(labelsFlag=True)
        return self
                
    
    #prediction step
    def predict(self, unitActivity):
        #given search query, predict probability of target feature
        #assuming target_feature is boolean for now (e.g. click_bool or booking_bool)
        unitActivity = cm.CUDAMatrix(unitActivity)
        
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
        v = cm.empty((self.numVisible, self.batchsize))
        vt = cm.CUDAMatrix(np.zeros((self.numClasses, self.batchsize)))
        h = cm.empty((self.numHidden, self.batchsize))
        r = cm.empty((self.numHidden, self.batchsize))
        idx = cm.CUDAMatrix(np.zeros((1, self.batchsize)))
        
        for batch in range(0,numBatches):
            if (batch + 1) % 10000 == 0:
                print "Batch #", batch+1
            if batch == numBatches - 1:
                batchsize = numTotalSamples - batchsize*(numBatches-1)
                data = unitActivity.slice((batch*self.batchsize), (batch*self.batchsize)+batchsize )
                v.assign(data)
            else:
                data = unitActivity.slice((batch*batchsize), ((batch+1)*batchsize) )
                v.assign(data)

            
            #probability method, target_feature represented as softmax (2 unit, pool)
            #up
            #ph = rm.logistic(gn.dot(self.W.T,data) + self.b) 
            cm.dot(self.W.T, v, target=h)
            h.add_col_vec(self.b)
            h.apply_sigmoid()
            
            #sample hiddens: nhstates = ph > gn.rand(self.numHidden, batchsize)
            r.fill_with_rand()
            r.less_than(h, target = h)
            
            #negclasses = rm.softmax(gn.dot(self.Wc, nhstates) + self.cc )
            #ignore for now: negclassesstates = rm.softmax_sample(negclasses)
            cm.dot(self.Wc, h, target=vt)
            vt.add_col_vec(self.cc)
            vt = self.softmax(vt)
            vt.argmax(axis=0, target=idx)
            
            if batch == numBatches - 1:
                labels[:,(batch*self.batchsize):(batch*self.batchsize)+batchsize] = idx.numpy_array
            else:
                labels[:,(batch*self.batchsize):((batch+1)*self.batchsize)] = idx.numpy_array

        end_time=time.time()
        print("Finished prediction step. Elapsed time was %g seconds" % (end_time - start_time))

        return labels




