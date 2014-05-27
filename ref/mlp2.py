"""
MLP using basic operations - version 2.

The activation and the cost function can be configured externally
in this version.  
 
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
    ''' Derivative of cross entropy with respect to the output ''' 
    return (outputs - targets) / (outputs * (1.0 - outputs)) 

def init_weights(nrows, ncols):
    return 0.01 * np.random.randn(nrows, ncols)

def error_rate(preds, labels):
    return 100.0 * np.mean(np.not_equal(preds, labels))

class MultilayerPerceptron:
    def fit(self, inputs, targets, nepochs, epsilon, nhidden, 
            g, gprime, de):
        ''' The extra parameters are:
                g       : the activation function.
                gprime  : derivative of the activation function with respect
                          to its inputs.
                de      : derivative of the cost function with respect
                          to the output of the network.
        '''
        nin = inputs.shape[1]
        nout = targets.shape[1]
    
        self.g = g
        self.gprime = gprime
        self.de = de
        self.weights1 = init_weights(nin, nhidden)
        self.weights2 = init_weights(nhidden, nout)

        for epoch in range(nepochs): 
            z1, hidden, z2, outputs = self.fprop(inputs)
            self.bprop(inputs, z1, hidden, z2, outputs, targets, epsilon)

    def predict(self, inputs):
        z1, hidden, z2, outputs = self.fprop(inputs)
        preds = np.argmax(outputs, axis=1) 
        return preds

    def fprop(self, inputs):
        z1 = np.dot(inputs, self.weights1)
        hidden = self.g(z1)
        z2 = np.dot(hidden, self.weights2)
        outputs = self.g(z2)
        return z1, hidden, z2, outputs

    def bprop(self, inputs, z1, hidden, z2, outputs, targets, epsilon):
        delta2 = self.de(outputs, targets) * self.gprime(z2) 
        delta1 = np.dot(delta2, self.weights2.T) * self.gprime(z1) 
        self.weights2 -= epsilon * np.dot(hidden.T, delta2)
        self.weights1 -= epsilon * np.dot(inputs.T, delta1)

if __name__ == '__main__':
    np.random.seed(0)
    trainData, unused1, trainTargets, testData, testLabels, unused2 = \
            cPickle.load(open('smnist.pkl'))
    net = MultilayerPerceptron()
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.0002, nhidden=50,
            g=logistic, gprime=logistic_prime, de=ce_de)
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
