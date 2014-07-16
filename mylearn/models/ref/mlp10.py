"""
MLP using basic operations - version 10.

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

class MultilayerPerceptron:
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
    net = MultilayerPerceptron()
    net.fit(trainData, trainTargets, nepochs=100, epsilon=0.1,
            mbs=100, loss=ce,
            confs=[(Type.fcon, logistic, 64),
                   (Type.fcon, logistic, trainTargets.shape[1])])
    
    preds = net.predict(testData)
    errorRate = error_rate(preds, testLabels)
    print 'test error rate ' + str(round(errorRate, 2)) + '%' 
