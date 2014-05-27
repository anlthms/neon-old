"""
MLP using basic operations - version 1.

This is a bare minimum implementation. Just one hidden layer and no bias inputs. 
Cross entropy is used as the cost function for training. All nodes have logistic
activation.

"""

import math
import cPickle
import numpy as np

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def init_weights(nrows, ncols):
    return 0.01 * np.random.randn(nrows, ncols)

def error_rate(preds, labels):
    return 100.0 * np.mean(np.not_equal(preds, labels))

class MultilayerPerceptron:
    def fit(self, inputs, targets, nepochs, epsilon, nhidden):
        nin = inputs.shape[1]
        nout = targets.shape[1]
    
        self.weights1 = init_weights(nin, nhidden)
        self.weights2 = init_weights(nhidden, nout)

        for epoch in range(nepochs): 
            hidden, outputs = self.fprop(inputs)
            self.bprop(inputs, hidden, outputs, targets, epsilon)

    def predict(self, inputs):
        hidden, outputs = self.fprop(inputs)
        preds = np.argmax(outputs, axis=1) 
        return preds

    def fprop(self, inputs):
        hidden = logistic(np.dot(inputs, self.weights1))
        outputs = logistic(np.dot(hidden, self.weights2))
        return hidden, outputs

    def bprop(self, inputs, hidden, outputs, targets, epsilon):
        delta2 = outputs - targets
        delta1 = np.dot(delta2, self.weights2.T) * hidden * (1 - hidden)
        self.weights2 -= epsilon * np.dot(hidden.T, delta2)
        self.weights1 -= epsilon * np.dot(inputs.T, delta1)

if __name__ == '__main__':
    np.random.seed(0)
    trainData, unused1, trainTargets, testData, testLabels, unused2 = \
            cPickle.load(open('smnist.pkl'))
    net = MultilayerPerceptron()
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.0002, nhidden=50)
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
