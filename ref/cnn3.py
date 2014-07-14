"""
CNN using basic operations - version 3.

Memory is preallocated for pre-activations, outputs and deltas of each layer.
 
"""

import cPickle
import numpy as np
from common import *

class Layer:
    def __init__(self, bs, nin, nout, g):
        self.weights = init_weights((nin, nout))
        self.g = g
        self.gprime = get_prime(g)
        self.nout = nout
        self.z = np.zeros((bs, nout))
        self.y = np.zeros((bs, nout))
        self.delta = np.zeros((bs, nout))
        
    def fprop(self, inputs):
        self.z[:] = np.dot(inputs, self.weights)
        self.y[:] = self.g(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.gprime(self.z) 

    def update(self, inputs, epsilon):
        self.weights -= epsilon * np.dot(inputs.T, self.delta)

    def resize(self, bs):
        self.z = np.zeros((bs, self.nout))
        self.y = np.zeros((bs, self.nout))
        self.delta = np.zeros((bs, self.nout))
        
    def error(self):
        return np.dot(self.delta, self.weights.T)

class ConvLayer:
    def __init__(self, bs, g, ishape, fshape, nfilt):
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
        self.z = np.zeros((bs, self.nout))
        self.y = np.zeros((bs, self.nout))
        self.delta = np.zeros((bs, self.nout))

        # Figure out the connections with the previous layer.   
        self.links = np.zeros((self.fmsize, self.fsize), dtype='i32')
        src = 0 # This variable tracks the top left corner
                # of the receptive field.
        for dst in range(self.fmsize):
            colinds = []
            # Collect the column indices for the
            # entire receptive field.
            for row in range(self.fheight):
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
        for i in range(self.nfilt):
            filt = self.weights[i]
            # Create a dense version of the weights with absent
            # links zeroed out and shared links duplicated.
            dweights = np.zeros((inputs.shape[1], self.fmsize))
            for dst in range(self.fmsize):
                dweights[self.links[dst], dst] = filt

            self.z[:, i * self.fmsize : (i + 1) * self.fmsize] = \
                    np.dot(inputs, dweights)

        self.y[:] = self.g(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.gprime(self.z) 

    def update(self, inputs, epsilon):
        for i in range(self.nfilt):
            wsums = np.zeros(self.weights[i].shape) 
            updates = np.dot(inputs.T, self.delta)
            for dst in range(self.fmsize):
                wsums += updates[self.links[dst], (i * self.fmsize + dst)]

            self.weights[i] -= epsilon * wsums

    def resize(self, bs):
        self.z = np.zeros((bs, self.nout))
        self.y = np.zeros((bs, self.nout))
        self.delta = np.zeros((bs, self.nout))

class Network:
    def fit(self, inputs, targets, nepochs, epsilon, loss, confs):
        nin = inputs.shape[1]
        self.loss = loss
        self.de = get_loss_de(loss) 
        self.nlayers = len(confs)
        self.layers = []
        for i in range(self.nlayers):
            layer = self.lcreate(inputs.shape[0], nin, confs[i])
            self.layers.append(layer)
            nin = layer.nout

        for epoch in range(nepochs): 
            self.fprop(inputs)
            self.bprop(targets)
            self.update(inputs, epsilon) 
            error = loss(self.layers[-1].y, targets)
            print 'epoch ' + str(epoch) + ' training error ' + \
                   str(round(error, 5))

    def predict(self, inputs):
        for i in range(self.nlayers):
            self.layers[i].resize(inputs.shape[0])
        outputs = self.fprop(inputs)
        preds = np.argmax(outputs, axis=1) 
        return preds

    def lcreate(self, bs, nin, conf):
        if conf[0] == Type.fcon:
            return Layer(bs, nin, nout=conf[2], g=conf[1])

        if conf[0] == Type.conv:
            return ConvLayer(bs, g=conf[1], ishape=conf[2],
                             fshape=conf[3], nfilt=conf[4])

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
    net = Network()
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.00008,
            loss=ce,
            confs=[(Type.conv, tanh, (28, 28), (5, 5), 2),
                   (Type.fcon, tanh, 64),
                   (Type.fcon, logistic, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
