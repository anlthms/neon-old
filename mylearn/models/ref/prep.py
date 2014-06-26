"""
Sample the MNIST dataset and create pickle files.
"""

import json
import cPickle
import numpy as np
import mnist_ubyte as mu

def load_mnist_data(path):
    # Training data.
    trainData = mu.read_mnist_images(path + 'train-images-idx3-ubyte' ,
                                     dtype='float32')
    trainData = trainData.reshape((60000, 784))

    # Sample 10% of the training data.
    inds = range(60000) 
    np.random.shuffle(inds)
    inds = inds[0:6000]
    trainData = trainData[inds]
    trainLabels = mu.read_mnist_labels(path + 'train-labels-idx1-ubyte')[inds]

    # Test data.
    testData = mu.read_mnist_images(path + 't10k-images-idx3-ubyte' ,
                                    dtype='float32')
    testData = np.reshape(testData, (10000, 784))
    testLabels = mu.read_mnist_labels(path + 't10k-labels-idx1-ubyte')

    # One-hot encoding
    trainTargets = np.zeros((6000, 10)) 
    for col in range(10):
        trainTargets[:, col] = trainLabels == col 

    testTargets = np.zeros((10000, 10)) 
    for col in range(10):
        testTargets[:, col] = testLabels == col 

    return trainData, trainLabels, trainTargets, testData, testLabels, \
           testTargets

if __name__ == '__main__':
    np.random.seed(0)
    settings = json.loads(open("settings.json").read())
    path = settings['mnist_path']
    ret = load_mnist_data(path)
    cPickle.dump(ret, open('smnist.pkl', 'wb'), -1)
