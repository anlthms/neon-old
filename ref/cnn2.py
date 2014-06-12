"""
CNN using basic operations - version 2.

Enhancements to make convolutions faster:
- Consider the convolutional layer as fully connected, but with null weights on
links that should not be connected. During weight updates, skip the links with
null weights.
- Initialize and update shared weights equally. 
 
"""

import math
import cPickle
import numpy as np
from common import *

class Type:
    fcon = 0    # Fully connected
    conv = 1    # Convolutional
    pool = 2    # Max-pooling

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
        self.weights = init_weights((nfilt, self.fsize))
        self.g = g
        self.gprime = get_prime(g)

        # Figure out the connections with the previous layer.   
        self.links = np.zeros((self.fmsize, self.fsize), dtype='i32')
        src = 0 # This variable tracks the top left corner
                # of the receptive field.
        for dst in range(self.fmsize):
            colinds = []
            for row in range(self.fheight):
                # Collect the column indices for the
                # entire receptive field.
                start = src + row * self.iwidth
                colinds += range(start, start + self.fwidth) 
            if (src % self.iwidth + self.fwidth) < self.iwidth:
                # Slide the filter by 1 cell.
                src += 1
            else:
                # We hit the right edge of the input image.
                # Sweep the filter over to the next row.
                src += self.fwidth
            self.links[dst] = colinds
        
    def fprop(self, inputs):
        self.z = np.zeros((inputs.shape[0], self.nout))
        for i in range(self.nfilt):
            filt = self.weights[i]
            # Create a dense version of the weights with absent
            # links zeroed out and shared links duplicated.
            dweights = np.zeros((inputs.shape[1], self.fmsize))
            for dst in range(self.fmsize):
                dweights[self.links[dst], dst] = filt

            self.z[:, i * self.fmsize : (i + 1) * self.fmsize] = \
                    np.dot(inputs, dweights)

        self.y = self.g(self.z)
        return self.y

    def bprop(self, nextlayer):
        self.delta = np.dot(nextlayer.delta, nextlayer.weights.T) * \
                     self.gprime(self.z) 

    def update(self, inputs, epsilon):
        for i in range(self.nfilt):
            wsums = np.zeros(self.weights[i].shape) 
            updates = np.dot(inputs.T, self.delta)
            for dst in range(self.fmsize):
                wsums += updates[self.links[dst], dst]

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
                   (Type.fcon, tanh, 64),
                   (Type.fcon, logistic, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
