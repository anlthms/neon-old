"""
MLP using basic operations - version 5.

Two hidden layers.

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

    def bprop(self, nextlayer):
        self.delta = np.dot(nextlayer.delta, nextlayer.weights.T) * \
                     self.gprime(self.z) 

    def update(self, inputs, epsilon):
        self.weights -= epsilon * np.dot(inputs.T, self.delta)

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
            self.bprop(inputs, targets)
            self.update(inputs, epsilon) 
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
        lastlayer.delta = self.de(lastlayer.y, targets) * \
                          lastlayer.gprime(lastlayer.z) 
        while i > 0:
            i -= 1 
            self.layers[i].bprop(self.layers[i + 1])

    def update(self, inputs, epsilon):
        self.layers[0].update(inputs, epsilon)
        for i in range(1, self.nlayers):
            self.layers[i].update(self.layers[i - 1].y, epsilon)

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
