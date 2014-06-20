"""
MLP using basic operations - version 8.

Softmax output layer. 

"""

import cPickle
import numpy as np
from common import * 

def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum(axis=1).reshape((ex.shape[0], 1))

class SoftmaxLayer(Layer):
    def __init__(self, nin, nout):
        self.weights = init_weights((nin, nout))
        self.g = softmax 
        self.nout = nout
        
    def bprop(self, error):
        self.delta = error * self.y * (1.0 - self.y) 

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
            error = loss(self.layers[-1].y, targets)
            print 'epoch ' + str(epoch) + ' training error ' + \
                   str(round(error, 5))

    def predict(self, inputs):
        outputs = self.fprop(inputs)
        preds = np.argmax(outputs, axis=1) 
        return preds

    def lcreate(self, nin, conf):
        if conf[0] == Type.fcon:
            if conf[1] == softmax:
                return SoftmaxLayer(nin + 1, nout=conf[2])
            else:
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
                   (Type.fcon, softmax, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
