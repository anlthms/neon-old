"""
CNN using basic operations - version 4.

Aggregated vector-matrix operations into matrix-matrix operations.
Minibatch training.
 
"""

import cPickle
import numpy as np
from common import *

class Layer:
    def __init__(self, nin, nout, g):
        self.weights = init_weights((nout, nin))
        self.g = g
        self.gprime = get_prime(g)
        self.nout = nout
        
    def fprop(self, inputs):
        self.z = np.dot(inputs, self.weights.T)
        self.y = self.g(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.gprime(self.z)

    def update(self, inputs, epsilon):
        self.weights -= epsilon * np.dot(self.delta.T, inputs)

    def error(self):
        return np.dot(self.delta, self.weights)

class ConvLayer:
    def __init__(self, g, ishape, fshape, nfilt):
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
        self.fmstarts = np.array(range(0, self.fmsize * self.nfilt,
                                       self.fmsize))
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
        self.z = np.zeros((inputs.shape[0], self.nout))
        for dst in range(self.fmsize):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            self.z[:, (self.fmstarts + dst)] = \
                    np.dot(inputs[:, self.links[dst]], self.weights.T)

        self.y = self.g(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.gprime(self.z) 

    def update(self, inputs, epsilon):
        wsums = np.zeros(self.weights.shape) 
        for dst in range(self.fmsize):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps. 
            wsums += np.dot(self.delta[:, (self.fmstarts + dst)].T,
                            inputs[:, self.links[dst]])
        # Update the filters after averaging the weight updates.
        self.weights -= epsilon * (wsums / self.fmsize) 

class Network:
    def fit(self, inputs, targets, nepochs, epsilon, mbs, loss, confs):
        nin = inputs.shape[1]
        self.loss = loss
        self.de = get_loss_de(loss) 
        self.nlayers = len(confs)
        self.layers = []
        self.mbs = mbs
        for i in range(self.nlayers):
            layer = self.lcreate(nin, confs[i])
            self.layers.append(layer)
            nin = layer.nout

        assert inputs.shape[0] % mbs == 0
        nbatch = inputs.shape[0] / mbs
        for epoch in range(nepochs): 
            error = 0
            for batch in range(nbatch):
                start = batch * mbs
                end = (batch + 1) * mbs
                self.fprop(inputs[start:end])
                self.bprop(targets[start:end])
                self.update(inputs[start:end], epsilon) 
                error += loss(self.layers[-1].y, targets[start:end])
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
            return ConvLayer(g=conf[1], ishape=conf[2],
                             fshape=conf[3], nfilt=conf[4])

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers: 
            y = layer.fprop(y)
        return y

    def bprop(self, targets):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        lastlayer.bprop(self.de(lastlayer.y, targets) / self.mbs)
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
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.08,
            mbs=100, loss=ce,
            confs=[(Type.conv, tanh, (28, 28), (5, 5), 2),
                   (Type.fcon, tanh, 64),
                   (Type.fcon, logistic, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
