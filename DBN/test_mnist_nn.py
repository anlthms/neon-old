import pickle
import numpy as np
import ipdb
import time
#import dbn_class as dc
import mnist_ubyte as mu
import cudamat as cm
import nn_class as nc

zscore = lambda x: (x - x.mean(axis=1)[:,np.newaxis]) / (x.std())

#mnist load code
trainData = mu.read_mnist_images('/usr/local/data/datasets/pylearn2/mnist/train-images-idx3-ubyte' , dtype='float32')
trainData = np.reshape(trainData, (60000,784)).T
#trainData = zscore(trainData) #for debugging relu only
trainLabels = mu.read_mnist_labels('/usr/local/data/datasets/pylearn2/mnist/train-labels-idx1-ubyte')
#trainLabels = int(trainLabels)

testData = mu.read_mnist_images('/usr/local/data/datasets/pylearn2/mnist/t10k-images-idx3-ubyte' , dtype='float32')
testData = np.reshape(testData, (10000,784)).T
#testData = zscore(testData) #for debugging relu only
testLabels = mu.read_mnist_labels('/usr/local/data/datasets/pylearn2/mnist/t10k-labels-idx1-ubyte')

#initialize NN with 1 hidden layer : 100 nodes
opts = dict()
opts['numFeatures'] = 784
opts['sizes'] = [2000, 1000, 1000]
#opts['sizes'] = [256, 256, 512, 512, 256, 256, 2048] #for investigation of different architectures
#opts['sizes'] = [512, 512, 2048]
opts['eta'] = .01 #.005
#opts['momentum'] = 0.5 #initial momentum (first 5 epochs)
opts['batchsize'] = 100
opts['maxEpoch'] = 1000
opts['FILE_LOAD_FLAG'] = False
opts['FROM_DBN_FLAG'] = True #if True use pre-trained DBN
LOAD_EXISTING_NN_MODEL = False

dbn = pickle.load(open('mnist_dbn.pkl'))

uniqueClasses = np.unique(trainLabels)
numClasses = len(uniqueClasses) #0-9 for mnist, assuming this is the format for labels for now
classes = np.zeros((numClasses , len(trainLabels)))
for i in range(len(trainLabels)):
    classes[int(trainLabels[i]), i] = 1.
    
opts['numClasses'] = numClasses

perfVec = [] #store performance across iterations

for iter in range(1): #for investigation of pipeline mode

    TEST_CROSSVALIDATION_PERFORMANCE = False

    if LOAD_EXISTING_NN_MODEL:
        #load pre-trained DBN model
        cm.cublas_init()
        cm.CUDAMatrix.init_random(1)
    
        nn = pickle.load(open('mnist_nn.pkl'))
        nn.maxEpoch = opts['maxEpoch']
       # nn.eta=.01
        #nn.actfunc='relu'
    else:
        nn = nc.nn_class(opts, dbn)
    
    #train DBN using CD
    #USE_ASSOCIATIVE_MEMORY_LABELS=True: use associative memory in N-1 layer
    #USE_ASSOCIATIVE_MEMORY_LABELS=False: pretrain for NN
    nn = nn.train(trainData[:,0:60000], classes[:,0:60000], resumeFlag=LOAD_EXISTING_NN_MODEL) 

    #have to figure out a way to pickle cudamat data
    pickle.dump(nn, open('mnist_nn.pkl', 'w'))

    #predict function for rbm_class
    #if TEST_CROSSVALIDATION_PERFORMANCE:
    # labels = dbn.test(trainData[:,50000:60000])
    # errFrac = float(np.sum(np.not_equal(labels, trainLabels[50000:60000])))/len(trainLabels[50000:60000])
    labels = nn.test(testData[:,0:10000])
    errFrac = float(np.sum(np.not_equal(labels, testLabels[0:10000])))/len(testLabels[0:10000])

    perfVec.append(errFrac)

    print  'Testing/Crossvalidation error= ' + str(errFrac)
    #else:
    labels = nn.test(trainData[:,0:60000])
    errFrac = float(np.sum(np.not_equal(labels, trainLabels[0:60000])))/len(trainLabels[0:60000])
    print  'Training error= ' + str(errFrac)

    
    
    LOAD_EXISTING_DBN_MODEL = True
    
    #cm.cublas_shutdown()

    #todo: visualize

print perfVec


#test with pre-trained network


