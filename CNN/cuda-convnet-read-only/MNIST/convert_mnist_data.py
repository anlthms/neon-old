#prepare MNIST data for cuda-convnet 
# 

import mnist_ubyte as mu
import cPickle
import numpy as np

#mnist load code
trainData = mu.read_mnist_images('/usr/local/data/datasets/pylearn2/mnist/train-images-idx3-ubyte' , dtype='float32')
trainData = np.reshape(trainData, (60000,784)).T
trainLabels = mu.read_mnist_labels('/usr/local/data/datasets/pylearn2/mnist/train-labels-idx1-ubyte')
#trainLabels = int(trainLabels)

data = []
filenames = []
labels = []

numBatches = 6
numSamplesPerBatch = 10000

for eachBatch in range(numBatches):
    print 'converting batch #', eachBatch+1
    out_file_name = 'data_batch_' + str(eachBatch+1)
    out_file = open(out_file_name,'w')
    data = trainData[:, numSamplesPerBatch*eachBatch:numSamplesPerBatch*(eachBatch+1)]
    labels = trainLabels[numSamplesPerBatch*eachBatch:numSamplesPerBatch*(eachBatch+1)]
    
    batch_label = 'batch ' + str(eachBatch+1) + ' of ' + str(numBatches+1)
    dic = {'batch_label':batch_label, 'data':data, 'labels':labels, 'filenames':filenames}
    cPickle.dump(dic, out_file)
    out_file.close()

testData = mu.read_mnist_images('/usr/local/data/datasets/pylearn2/mnist/t10k-images-idx3-ubyte' , dtype='float32')
testData = np.reshape(testData, (10000,784)).T
testLabels = mu.read_mnist_labels('/usr/local/data/datasets/pylearn2/mnist/t10k-labels-idx1-ubyte')

eachBatch = numBatches #test batch
print 'converting batch #', eachBatch+1
out_file_name = 'data_batch_' + str(eachBatch+1)
out_file = open(out_file_name,'w')
data = testData
labels = testLabels

batch_label = 'batch ' + str(eachBatch+1) + ' of ' + str(numBatches+1)
dic = {'batch_label':batch_label, 'data':data, 'labels':labels, 'filenames':filenames}
cPickle.dump(dic, out_file)
out_file.close()

#prepare the batches.meta file
#some info on format at URL below, but we need additional parameters
#http://www.cs.toronto.edu/~kriz/cifar.html
out_file = open('batches.meta','w')
label_names = dict()
for eachClass in range(10):
    label_names[eachClass] = eachClass
num_cases_per_batch = 10000
num_vis = 28**2
data_mean = np.mean(trainData, axis=1)[:,np.newaxis]
dic = { 'num_cases_per_batch': num_cases_per_batch, 'label_names': label_names, 'num_vis': num_vis, 'data_mean': data_mean}
cPickle.dump(dic, out_file)
out_file.close()


