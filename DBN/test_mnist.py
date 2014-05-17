import pickle
import numpy as np
import ipdb
import time
import dbn_class as dc
import mnist_ubyte as mu
import cudamat as cm
#import nn_class as nc

zscore = lambda x: (x - x.mean(axis=1)[:,np.newaxis]) / (x.std())

#mnist load code
trainData = mu.read_mnist_images('/usr/local/data/datasets/pylearn2/mnist/train-images-idx3-ubyte' , dtype='float32')
trainData = np.reshape(trainData, (60000,784)).T
trainData = zscore(trainData) #for debugging relu only
trainLabels = mu.read_mnist_labels('/usr/local/data/datasets/pylearn2/mnist/train-labels-idx1-ubyte')
#trainLabels = int(trainLabels)

#extremely inefficient way to build a balanced dataset for each minibatch
BALANCE_DATA_FLAG = False
if BALANCE_DATA_FLAG:
    print "balancing training data"
    newTrainData = np.zeros((784, 60000))
    newTrainLabels = np.zeros((60000))
    classPtr = -1*np.ones(10)
    for i in range(6000):
        #print i
        for classLabel in range(10):
            seekPtr = 0
            if classPtr[classLabel] == -1:
                seekPtr += 1
                while trainLabels[classPtr[classLabel] + seekPtr] != classLabel:
                    seekPtr += 1
                classPtr[classLabel] = classPtr[classLabel] + seekPtr #classPtr now points to first occurence of classLabel 
            if trainLabels[classPtr[classLabel]] != classLabel:
                while trainLabels[classPtr[classLabel] + seekPtr] != classLabel:
                    seekPtr += 1
                    if classPtr[classLabel] + seekPtr == 60000:
                        classPtr[classLabel] = 0
                        seekPtr = 0
                    
                classPtr[classLabel] = classPtr[classLabel] + seekPtr #classPtr now points to next occurence of classLabel 
        
            newTrainData[:,i*10 + classLabel] = trainData[:,classPtr[classLabel]]
            newTrainLabels[i*10 + classLabel] = classLabel
            classPtr[classLabel] += 1
            if classPtr[classLabel] == 60000:
                classPtr[classLabel] = 0
        
    print "done balancing training data."        
    #ipdb.set_trace()
    trainData = newTrainData
    trainLabels = newTrainLabels

testData = mu.read_mnist_images('/usr/local/data/datasets/pylearn2/mnist/t10k-images-idx3-ubyte' , dtype='float32')
testData = np.reshape(testData, (10000,784)).T
testData = zscore(testData) #for debugging relu only
testLabels = mu.read_mnist_labels('/usr/local/data/datasets/pylearn2/mnist/t10k-labels-idx1-ubyte')

#ipdb.set_trace()

#initialize DBN with 1 hidden layer : 100 nodes
opts = dict()
opts['numFeatures'] = 784
#opts['sizes'] = [800,800,800,800]
#opts['sizes'] = [256, 256, 512, 512, 256, 256, 2048] #for investigation of different architectures
#opts['sizes'] = [256, 256, 512]
opts['sizes'] = [2000, 1000, 1000]
opts['eta'] = 0.005 #.005
opts['momentum'] = 0. #initial momentum (first 5 epochs)
opts['batchsize'] = 100
opts['maxEpoch'] = 20

LOAD_EXISTING_DBN_MODEL = False
USE_ASSOCIATIVE_MEMORY_LABELS=False #change to True if classifying using DBN; False if using to pretrain for a DNN

perfVec = [] #store performance across iterations

for iter in range(1): #for investigation of pipeline mode

    TEST_CROSSVALIDATION_PERFORMANCE = False

    if LOAD_EXISTING_DBN_MODEL:
        #load pre-trained DBN model
        cm.cublas_init()
        cm.CUDAMatrix.init_random(1)
    
        dbn = pickle.load(open('mnist_dbn.pkl'))
        numRBMs = len(dbn.rbm)
        for eachRBM in range(numRBMs):
            dbn.rbm[eachRBM].maxEpoch = opts['maxEpoch']
        #if using different maxEpoch for different RBMs
        #dbn.rbm[0].maxEpoch = 0
        #dbn.rbm[1].maxEpoch = 5
        #dbn.rbm[2].maxEpoch = 5
    else:
        dbn = dc.dbn_class(opts)
    
    #train DBN using CD
    #USE_ASSOCIATIVE_MEMORY_LABELS=True: use associative memory in N-1 layer
    #USE_ASSOCIATIVE_MEMORY_LABELS=False: pretrain for NN
    dbn = dbn.train(trainData[:,0:60000], trainLabels[0:60000], USE_ASSOCIATIVE_MEMORY_LABELS=USE_ASSOCIATIVE_MEMORY_LABELS, resumeFlag=LOAD_EXISTING_DBN_MODEL)

    #have to figure out a way to pickle cudamat data
    pickle.dump(dbn, open('mnist_dbn.pkl', 'w'))
    
    if USE_ASSOCIATIVE_MEMORY_LABELS:
    
        #predict function for rbm_class
        #if TEST_CROSSVALIDATION_PERFORMANCE:
        # labels = dbn.test(trainData[:,50000:60000])
        # errFrac = float(np.sum(np.not_equal(labels, trainLabels[50000:60000])))/len(trainLabels[50000:60000])
        labels = dbn.test(testData[:,0:10000])
        errFrac = float(np.sum(np.not_equal(labels, testLabels[0:10000])))/len(testLabels[0:10000])

        perfVec.append(errFrac)

        print  'Testing/Crossvalidation error= ' + str(errFrac)
        #else:
        labels = dbn.test(trainData[:,0:60000])
        errFrac = float(np.sum(np.not_equal(labels, trainLabels[0:60000])))/len(trainLabels[0:60000])
        print  'Training error= ' + str(errFrac)

    
    
    LOAD_EXISTING_DBN_MODEL = True
    
    #cm.cublas_shutdown()

    #todo: visualize

if USE_ASSOCIATIVE_MEMORY_LABELS:
    print perfVec


#unroll to nn and train with backprop


