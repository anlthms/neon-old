"""
MLP using basic operations - version 7.

Added bias inputs. 
 
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

def get_loss_de(func):
    if func == ce:
        return ce_de

def ce(outputs, targets):
    return np.mean(-targets * np.log(outputs) - \
                   (1 - targets) * np.log(1 - outputs))

def ce_de(outputs, targets):
    return (outputs - targets) / (outputs * (1.0 - outputs)) 

def init_weights(shape):
    return np.random.uniform(-0.1, 0.1, shape)

def error_rate(preds, labels):
    return 100.0 * np.mean(np.not_equal(preds, labels))

def append_bias(data):
    """ Append a column of ones. """
    return np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)

class Type:
    fcon = 0    # Fully connected

class Layer:
    def __init__(self, nin, nout, g):
        self.weights = init_weights((nin, nout))
        self.g = g
        self.gprime = get_prime(g)
        self.nout = nout
        
    def fprop(self, inputs):
        inputs = append_bias(inputs)
        self.z = np.dot(inputs, self.weights)
        self.y = self.g(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.gprime(self.z)

    def update(self, inputs, epsilon):
        inputs = append_bias(inputs)
        self.weights -= epsilon * np.dot(inputs.T, self.delta)

    def error(self):
        """ Omit the bias column from the weights matrix. """
        return np.dot(self.delta, self.weights[:-1,:].T)

class MultilayerPerceptron:
    def fit(self, inputs, targets, nepochs, epsilon, loss, confs):
        nin = inputs.shape[1]
        self.loss = loss
        self.de = get_loss_de(loss) 
        self.nlayers = len(confs)
        self.layers = []
        for i in range(self.nlayers):
            layer = self.lcreate(nin, confs[i])
            self.layers.append(layer)
            nin = layer.nout

        for epoch in range(nepochs): 
            self.fprop(inputs)
            self.bprop(targets)
            self.update(inputs, epsilon) 

    def predict(self, inputs):
        outputs = self.fprop(inputs)
        preds = np.argmax(outputs, axis=1) 
        return preds

    def lcreate(self, nin, conf):
        if conf[0] == Type.fcon:
            # Add 1 for the bias input.
            return Layer(nin + 1, nout=conf[2], g=conf[1])

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers: 
            y = layer.fprop(y)
        return y

    def bprop(self, targets):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        lastlayer.bprop(self.de(lastlayer.y, targets))
        while i > 0:
            error = self.layers[i].error()
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
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.0001,
            loss=ce,
            confs=[(Type.fcon, logistic, 64),
                   (Type.fcon, logistic, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
