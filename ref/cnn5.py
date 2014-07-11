"""
CNN using basic operations - version 5.

Added a maxpooling layer.
 
"""

import cPickle
import numpy as np
from common import *

class Layer:
    def __init__(self, mbs, nin, nout, g):
        self.weights = init_weights((nout, nin))
        self.g = g
        self.gprime = get_prime(g)
        self.nout = nout
        self.z = np.zeros((mbs, nout))
        self.y = np.zeros((mbs, nout))
        self.delta = np.zeros((mbs, nout))
        
    def fprop(self, inputs):
        self.z[:] = np.dot(inputs, self.weights.T)
        self.y[:] = self.g(self.z)

    def bprop(self, error):
        self.delta[:] = error * self.gprime(self.z)

    def update(self, inputs, epsilon):
        self.weights -= epsilon * np.dot(self.delta.T, inputs)

    def error(self):
        return np.dot(self.delta, self.weights)

class MaxpoolLayer:
    def __init__(self, mbs, g, nfm, ifmshape, pshape):
        """
            g        : Activation function
            nfm      : Number of input feature maps
            ifmshape : Shape of the input feaure map  
            pshape   : Pooling shape
        """
        self.g = g
        self.nfm = nfm
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.pheight, self.pwidth = pshape
        self.psize = self.pheight * self.pwidth
        assert self.ifmheight % self.pheight == 0
        assert self.ifmwidth % self.pwidth == 0

        self.ofmsize = self.ifmsize / self.psize
        self.gprime = get_prime(g)
        self.nin = nfm * self.ifmsize
        self.nout = nfm * self.ofmsize 
        self.z = np.zeros((mbs, self.nout))
        self.y = np.zeros((mbs, self.nout))
        self.delta = np.zeros((mbs, self.nout))
        self.maxinds = np.zeros((mbs * nfm, self.ofmsize), dtype='i32')
        self.berror = np.zeros((mbs, self.nin)) # This is the error matrix
                                                # associated with the previous
                                                # layer.
        # Figure out the possible connections with the previous layer.   
        # Each unit in this layer could be connected to any one of
        # self.psize units in the previous layer. 
        self.links = np.zeros((self.ofmsize, self.psize), dtype='i32')
        src = 0 # This variable tracks the top left corner
                # of the receptive field.
        for dst in range(self.ofmsize):
            colinds = []
            # Collect the column indices for the
            # entire receptive field.
            for row in range(self.pheight):
                start = src + row * self.ifmwidth
                colinds += range(start, start + self.pwidth) 
            src += self.pwidth 
            self.links[dst] = colinds

    def fprop(self, inputs):
        # Reshape the input so that we have a separate row
        # for each input feature map (this is to avoid a loop over
        # each feature map).
        inputs = squish(inputs, self.nfm) 
        rz = squish(self.z, self.nfm)
        for dst in range(self.ofmsize):
            # For this output unit, get the corresponding receptive fields
            # within all input feature maps.
            rf = inputs.take(self.links[dst], axis=1)
            # Save the index of the maximum value within the receptive fields.
            self.maxinds[:, dst] = rf.argmax(axis=1)
            # Set the output to the maximum value.
            rz[:, dst] = rf[range(rf.shape[0]), self.maxinds[:, dst]]
            
        # Reshape back to original shape.
        self.z[:] = rz.reshape((self.z.shape))
        self.y[:] = self.g(self.z)

    def bprop(self, error):
        self.delta[:] = error * self.gprime(self.z) 

    def update(self, inputs, epsilon):
        # There are no weights to update.
        pass

    def error(self):
        self.berror[:] = np.zeros(self.berror.shape)
        # Reshape the backpropagated error matrix to have one
        # row per feature map. 
        rberror = squish(self.berror, self.nfm)
        # Reshape the delta matrix to have one row per feature map.
        rdelta = squish(self.delta, self.nfm)
        for dst in range(self.ofmsize):
            inds = self.links[dst][self.maxinds[:, dst]]
            rberror[range(rberror.shape[0]), inds] = rdelta[:, dst]
            
        self.berror[:] = rberror.reshape(self.berror.shape)
        return self.berror

class ConvLayer:
    def __init__(self, mbs, g, nifm, ifmshape, fshape, nfilt):
        self.ifmheight, self.ifmwidth = ifmshape 
        self.fheight, self.fwidth = fshape

        ofmheight = self.ifmheight - self.fheight + 1
        ofmwidth = self.ifmwidth - self.fwidth + 1
        self.ofmsize = ofmheight * ofmwidth
        self.nout = self.ofmsize * nfilt

        self.nifm = nifm
        self.nfilt = nfilt
        self.fsize = nifm * self.fheight * self.fwidth 
        self.weights = init_weights((nfilt, self.fsize))
        self.g = g
        self.gprime = get_prime(g)
        self.z = np.zeros((mbs, self.nout))
        self.y = np.zeros((mbs, self.nout))
        self.delta = np.zeros((mbs, self.nout))
        self.berror = np.zeros((mbs, self.ifmheight * self.ifmwidth * nifm))
        self.ofmstarts = np.array(range(0, (self.ofmsize * nfilt), self.ofmsize))
        # Figure out the connections with the previous layer.   
        self.links = np.zeros((self.ofmsize, self.fsize), dtype='i32')
        src = 0 # This variable tracks the top left corner
                # of the receptive field.
        for dst in range(self.ofmsize):
            colinds = []
            for row in range(self.fheight):
                # Collect the column indices for the
                # entire receptive field.
                start = src + row * self.ifmwidth
                for ifm in range(nifm):
                    colinds += range(start + ifm * self.ifmwidth,
                                     start + self.fwidth + ifm * self.ifmwidth) 
            if (src % self.ifmwidth + self.fwidth) < self.ifmwidth:
                # Slide the filter by 1 cell.
                src += 1
            else:
                # We hit the right edge of the input image.
                # Sweep the filter over to the next row.
                src += self.fwidth
            self.links[dst] = colinds

    def fprop(self, inputs):
        for dst in range(self.ofmsize):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            self.z[:, (self.ofmstarts + dst)] = \
                    np.dot(inputs.take(self.links[dst], axis=1),
                           self.weights.T)
        self.y[:] = self.g(self.z)

    def bprop(self, error):
        self.delta[:] = error * self.gprime(self.z) 

    def update(self, inputs, epsilon):
        wsums = np.zeros(self.weights.shape) 
        for dst in range(self.ofmsize):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps. 
            wsums += np.dot(self.delta.take((self.ofmstarts + dst), axis=1).T,
                            inputs.take(self.links[dst], axis=1))
        # Update the filters after averaging the weight updates.
        self.weights -= epsilon * (wsums / self.ofmsize) 

class Network:
    def fit(self, inputs, targets, nepochs, epsilon, mbs, loss, confs):
        nin = inputs.shape[1]
        self.loss = loss
        self.de = get_loss_de(loss) 
        self.nlayers = len(confs)
        self.layers = []
        self.mbs = mbs
        for i in range(self.nlayers):
            layer = self.lcreate(mbs, nin, confs[i])
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
        assert inputs.shape[0] % self.mbs == 0
        nbatch = inputs.shape[0] / self.mbs
        outputs = np.zeros((inputs.shape[0], self.layers[-1].nout)) 
        for batch in range(nbatch):
            start = batch * self.mbs
            end = (batch + 1) * self.mbs
            self.fprop(inputs[start:end])
            outputs[start:end] = self.layers[-1].y
        preds = np.argmax(outputs, axis=1) 
        return preds

    def lcreate(self, mbs, nin, conf):
        if conf[0] == Type.fcon:
            return Layer(mbs, nin, nout=conf[2], g=conf[1])

        if conf[0] == Type.conv:
            return ConvLayer(mbs, g=conf[1], nifm=conf[2], ifmshape=conf[3],
                             fshape=conf[4], nfilt=conf[5])
        if conf[0] == Type.pool:
            return MaxpoolLayer(mbs, g=conf[1], nfm=conf[2],
                                ifmshape=conf[3], pshape=conf[4])

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers: 
            layer.fprop(y)
            y = layer.y

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
            # The format of the configuration tuple for a convolution layer is:
            # (layer type, activation type, number of input feature maps,
            # shape of input feature map, filter shape, number of filters).
            #
            # The format of the configuration tuple for a maxpooling layer is:
            # (layer type, activation type, number of input feature maps,
            # shape of input feature map, pooling shape).
            confs=[(Type.conv, tanh, 1, (28, 28), (5, 5), 2),
                   (Type.pool, tanh, 2, (24, 24), (2,2)),
                   (Type.fcon, tanh, 64),
                   (Type.fcon, logistic, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
