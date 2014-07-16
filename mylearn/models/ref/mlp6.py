"""
MLP using basic operations - version 6.

Update the weights as early as possible - do not wait for the errors
to be propagated all the way back. 

"""

import math
import cPickle
import numpy as np
from common import *

class Type:
    fcon = 0    # Fully connected

class Layer:
    def __init__(self, nin, nout, g):
        self.weights = init_weights((nin, nout))
        self.g = g
        self.gprime = get_prime(g)
        self.nout = nout
        
    def fprop(self, inputs):
        self.z = np.dot(inputs, self.weights)
        self.y = self.g(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.gprime(self.z) 

    def update(self, inputs, epsilon):
        self.weights -= epsilon * np.dot(inputs.T, self.delta)

    def error(self):
        return np.dot(self.delta, self.weights.T)

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
            self.bprop(inputs, targets, epsilon)
            error = loss(self.layers[-1].y, targets)
            print 'epoch ' + str(epoch) + ' training error ' + \
                   str(round(error, 5))

    def predict(self, inputs):
        outputs = self.fprop(inputs)
        preds = np.argmax(outputs, axis=1) 
        return preds

    def lcreate(self, nin, conf):
        if conf[0] == Type.fcon:
            return Layer(nin, nout=conf[2], g=conf[1])

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers: 
            y = layer.fprop(y)
        return y

    def bprop(self, inputs, targets):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        lastlayer.bprop(self.de(lastlayer.y, targets))
        while i > 0:
            error = self.layers[i].error()
            i -= 1 
            self.layers[i].bprop(error)

    def bprop(self, inputs, targets, epsilon):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        lastlayer.bprop(self.de(lastlayer.y, targets))
        while i > 0:
            error = self.layers[i].error()
            i -= 1 
            self.layers[i].bprop(error)
            self.layers[i + 1].update(self.layers[i].y, epsilon)

        self.layers[i].update(inputs, epsilon)

if __name__ == '__main__':
    np.random.seed(0)
    trainData, unused1, trainTargets, testData, testLabels, unused2 = \
            cPickle.load(open('smnist.pkl'))
    net = MultilayerPerceptron()
    net.fit(trainData, trainTargets, nepochs=600, epsilon=0.0001,
            loss=ce,
            confs=[(Type.fcon, logistic, 100),
                   (Type.fcon, logistic, 64),
                   (Type.fcon, logistic, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
