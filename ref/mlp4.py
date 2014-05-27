"""
MLP using basic operations - version 4.

Added a separate "Layer" class and refactored the code.
 
"""

import math
import cPickle
import numpy as np

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_prime(z):
    y = logistic(z)
    return y * (1.0 - y) 

def get_prime(func):
    if func == logistic:
        return logistic_prime

def ce_de(outputs, targets):
    return (outputs - targets) / (outputs * (1.0 - outputs)) 

def init_weights(nrows, ncols):
    return 0.01 * np.random.randn(nrows, ncols)

def error_rate(preds, labels):
    return 100.0 * np.mean(np.not_equal(preds, labels))

class Layer:
    def __init__(self, nin, nout, g):
        self.weights = init_weights(nin, nout)
        self.g = g
        self.gprime = get_prime(g)
        
    def fprop(self, inputs):
        self.z = np.dot(inputs, self.weights)
        self.y = self.g(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.gprime(self.z)

    def update(self, inputs, epsilon):
        self.weights -= epsilon * np.dot(inputs.T, self.delta)

class MultilayerPerceptron:
    def fit(self, inputs, targets, nepochs, epsilon,
            nhidden, g, de):
        nunits = [inputs.shape[1]] + nhidden + [targets.shape[1]]
    
        self.de = de
        self.nlayers = len(nhidden) + 1
        self.layers = [Layer(nunits[i], nunits[i + 1], g[i])
                       for i in range(self.nlayers)]
        for epoch in range(nepochs): 
            self.fprop(inputs)
            self.bprop(inputs, targets)
            self.update(inputs, epsilon) 

    def predict(self, inputs):
        outputs = self.fprop(inputs)
        preds = np.argmax(outputs, axis=1) 
        return preds

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers: 
            y = layer.fprop(y)
        return y

    def bprop(self, inputs, targets):
        i = self.nlayers - 1
        error = self.de(self.layers[i].y, targets)
        self.layers[i].bprop(error)
        while i > 0:
            error = np.dot(self.layers[i].delta, self.layers[i].weights.T) 
            i -= 1 
            self.layers[i].bprop(error)

    def update(self, inputs, epsilon):
        self.layers[0].update(inputs, epsilon)
        for i in range(1, self.nlayers):
            self.layers[i].update(self.layers[i - 1].y, epsilon)

if __name__ == '__main__':
    np.random.seed(0)
    trainData, unused1, trainTargets, testData, testLabels, unused2 = \
            cPickle.load(open('smnist.pkl'))
    net = MultilayerPerceptron()
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.0002,
            nhidden=[50], g=[logistic, logistic], de=ce_de)
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
