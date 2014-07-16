"""
MLP using basic operations - version 3.

Added support for multiple hidden layers. Each layer can be configured
with its own activation function.
 
"""

import math
import cPickle
import numpy as np

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_prime(z):
    y = logistic(z)
    return y * (1.0 - y) 

def ce_de(outputs, targets):
    return (outputs - targets) / (outputs * (1.0 - outputs)) 

def init_weights(shape):
    return np.random.uniform(-0.1, 0.1, shape)

def error_rate(preds, labels):
    return 100.0 * np.mean(np.not_equal(preds, labels))

class MultilayerPerceptron:
    def fit(self, inputs, targets, nepochs, epsilon,
            nhidden, g, gprime, de):
        nin = inputs.shape[1]
        nout = targets.shape[1]
    
        self.g = g
        self.gprime = gprime
        self.de = de
        self.nlayers = len(nhidden) + 1
        self.weights = [init_weights((i, j))
                        for i, j in zip([nin] + nhidden, nhidden + [nout])]

        for epoch in range(nepochs): 
            results = self.fprop(inputs)
            self.bprop(inputs, results, targets, epsilon)

    def predict(self, inputs):
        results = self.fprop(inputs)
        outputs = results[-1][1]
        preds = np.argmax(outputs, axis=1) 
        return preds

    def fprop(self, inputs):
        results = [] 
        y = inputs
        for layer in range(self.nlayers): 
            z = np.dot(y, self.weights[layer])
            y = self.g[layer](z)
            results.append((z, y))

        return results

    def bprop(self, inputs, results, targets, epsilon):
        # Compute the deltas for each layer.
        layer = self.nlayers - 1
        z, y = results[layer]
        deltas = [self.de(y, targets) * self.gprime[layer](z)]
        while layer > 0:
            layer -= 1 
            z = results[layer][0]
            deltas.insert(0, np.dot(deltas[0], self.weights[layer + 1].T) * \
                          self.gprime[layer](z))

        # Update the weights.
        self.weights[0] -= epsilon * np.dot(inputs.T, deltas[0])
        for layer in range(1, self.nlayers):
            y = results[layer - 1][1]
            self.weights[layer] -= epsilon * np.dot(y.T, deltas[layer])

if __name__ == '__main__':
    np.random.seed(0)
    trainData, unused1, trainTargets, testData, testLabels, unused2 = \
            cPickle.load(open('smnist.pkl'))
    net = MultilayerPerceptron()
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.0001,
            nhidden=[64], g=[logistic, logistic],
            gprime=[logistic_prime, logistic_prime], de=ce_de)
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
