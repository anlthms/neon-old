import pickle
import numpy as np
import ipdb
import time
import dbn_class as dc
import mnist_ubyte as mu

#mnist load code
trainData = mu.read_mnist_images('/usr/local/data/datasets/pylearn2/mnist/train-images-idx3-ubyte' , dtype='float32')
trainData = np.reshape(trainData, (60000,784)).T
trainLabels = mu.read_mnist_labels('/usr/local/data/datasets/pylearn2/mnist/train-labels-idx1-ubyte')

#initialize DBN with 1 hidden layer : 100 nodes
opts = dict()
opts['numFeatures'] = 784
opts['sizes'] = [100, 100]
opts['eta'] = 0.01
opts['momentum'] = 0.9
opts['batchsize'] = 100
opts['maxEpoch'] = 1

LOAD_EXISTING_DBN_MODEL = False

if LOAD_EXISTING_DBN_MODEL:
    #load pre-trained DBN model
    dbn = pickle.load(open('mnist_dbn.pkl'))
else:
    dbn = dc.dbn_class(opts)
    
#train DBN using CD
dbn = dbn.train(trainData, trainLabels, LOAD_EXISTING_DBN_MODEL)
pickle.dump(dbn, open('mnist_dbn.pkl', 'w'))

#predict function for rbm_class
labels = dbn.test(trainData)

errFrac = float(np.sum(np.not_equal(labels, trainLabels)))/len(trainLabels)
print  'Training error= ' + str(errFrac)

#todo: cuda class for rbm_class & rbm_math
#todo: visualize
#todo: shuffle batches

#todo: andrew's questions regarding relu and output
