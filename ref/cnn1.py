"""
CNN using basic operations - version 1.
 
"""

import math
import cPickle
import numpy as np

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_prime(z):
    y = logistic(z)
    return y * (1.0 - y) 

def tanh(z):
    y = np.exp(-2 * z)
    return  (1.0 - y) / (1.0 + y)

def tanh_prime(z):
    y = tanh(z)
    return 1.0 - y * y

def get_prime(func):
    if func == logistic:
        return logistic_prime
    if func == tanh:
        return tanh_prime

def get_loss_de(func):
    if func == ce:
        return ce_de

def ce(outputs, targets):
    return np.mean(-targets * np.log(outputs) - \
                   (1 - targets) * np.log(1 - outputs))

def ce_de(outputs, targets):
    return (outputs - targets) / (outputs * (1.0 - outputs)) 

def init_weights(nrows, ncols):
    return 0.1 * np.random.randn(nrows, ncols)

def error_rate(preds, labels):
    return 100.0 * np.mean(np.not_equal(preds, labels))

class Type:
    fcon = 0    # Fully connected
    conv = 1    # Convolutional
    pool = 2    # Max-pooling

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

class ConvLayer:
    def __init__(self, nin, g, ishape, fshape, nfilt):
        self.iheight, self.iwidth = ishape 
        self.fheight, self.fwidth = fshape

        fmheight = self.iheight - self.fheight + 1
        fmwidth = self.iwidth - self.fwidth + 1
        self.fmsize = fmheight * fmwidth
        self.nout = self.fmsize * nfilt

        self.nfilt = nfilt
        self.fsize = self.fheight * self.fwidth 
        self.weights = init_weights(nfilt, self.fsize)
        self.g = g
        self.gprime = get_prime(g)
        
    def fprop(self, inputs):
        self.z = np.zeros((inputs.shape[0], self.nout))
        for i in range(self.nfilt):
            filt = self.weights[i]
            src = 0 # This variable tracks the top left corner
                    # of the receptive field.
            for dst in range(self.fmsize):
                # This loop can be replaced with a call
                # to scipy.signal.fftconvolve.
                colinds = []
                for row in range(self.fheight):
                    # Collect the column indices for the
                    # entire receptive field.
                    start = src + row * self.iwidth
                    colinds += range(start, start + self.fwidth) 

                # Compute the weighted average of the receptive field
                # and store the result within the destination feature map.
                self.z[:, (i * self.fmsize + dst)] = \
                        np.dot(inputs[:, colinds], filt)
                if (src % self.iwidth + self.fwidth) >= self.iwidth:
                    # We hit the right edge of the input image.
                    # Sweep the filter over to the next row.
                    src += self.fwidth
                else:
                    # Slide the filter by 1 cell.
                    src += 1

        self.y = self.g(self.z)
        return self.y

    def bprop(self, nextlayer):
        self.delta = np.dot(nextlayer.delta, nextlayer.weights.T) * \
                     self.gprime(self.z) 

    def update(self, inputs, epsilon):
        for i in range(self.nfilt):
            wsums = np.zeros(self.weights[i].shape) 
            src = 0
            for dst in range(self.fmsize):
                # This loop can be replaced with a call
                # to scipy.signal.fftconvolve.
                colinds = []
                for row in range(self.fheight):
                    start = src + row * self.iwidth
                    colinds += range(start, start + self.fwidth) 
                wsums += np.dot(inputs[:, colinds].T, self.delta[:, dst])
                if (src % self.iwidth + self.fwidth) >= self.iwidth:
                    src += self.fwidth
                else:
                    src += 1
            self.weights[i] -= epsilon * (wsums / self.fmsize) 

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

        if conf[0] == Type.conv:
            return ConvLayer(nin, g=conf[1], ishape=conf[2],
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
    net = Network()
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.00008,
            loss=ce,
            confs=[(Type.conv, tanh, (28, 28), (5, 5), 2),
                   (Type.fcon, tanh, 50),
                   (Type.fcon, logistic, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
