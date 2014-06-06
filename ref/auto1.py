"""
Autoencoder using basic operations - version 1.

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
    if func == sse:
        return sse_de
    if func == ce:
        return ce_de
    
def sse(outputs, targets):
    """ Sum of squared errors """
    return np.sum((outputs - targets) ** 2)

def sse_de(outputs, targets):
    """ Derivative of SSE with respect to the output """
    return (outputs - targets)

def ce(outputs, targets):
    return np.mean(-targets * np.log(outputs) - \
                   (1 - targets) * np.log(1 - outputs))

def ce_de(outputs, targets):
    return (outputs - targets) / (outputs * (1.0 - outputs)) 

def init_weights(nrows, ncols):
    return 0.01 * np.random.randn(nrows, ncols)

def error_rate(preds, labels):
    return 100.0 * np.mean(np.not_equal(preds, labels))

class Type:
    fcon = 0    # Fully connected

class Layer:
    def __init__(self, nin, nout, g):
        self.weights = init_weights(nin, nout)
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

class Network:
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
            error = self.loss(self.layers[-1].y, targets)
            print 'epoch ' + str(epoch) + ' training error ' + \
                   str(round(error, 5))

    def predict(self, inputs):
        outputs = self.fprop(inputs)
        preds = np.argmax(outputs, axis=1) 
        return preds

    def lcreate(self, nin, conf):
        if conf[0] == Type.fcon:
            return Layer(nin, nout=conf[2], g=conf[1])

        if conf[0] == Type.conv:
            return ConvLayer(nin, g=conf[1], ishape = conf[2],
                             fshape=conf[3], nfilt=conf[4])

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

    # Train the autoencoder first.
    auto = Network()
    alldata = np.vstack((trainData, testData))
    auto.fit(alldata, alldata, nepochs=200, epsilon=0.00004, loss=sse,
             confs=[(Type.fcon, logistic, 600),
                    (Type.fcon, logistic, alldata.shape[1])])
    
    trainCodes = auto.layers[0].y[0:trainData.shape[0]]
    testCodes = auto.layers[0].y[trainData.shape[0]:alldata.shape[0]]

    # Now classify.
    mlp = Network()
    mlp.fit(trainCodes, trainTargets, nepochs=100, epsilon=0.0002, loss=ce,
            confs=[(Type.fcon, logistic, 50),
                   (Type.fcon, logistic, trainTargets.shape[1])])

    preds = mlp.predict(testCodes)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
