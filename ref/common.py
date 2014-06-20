import numpy as np
import math as mt

class Type:
    fcon = 0    # Fully connected
    conv = 1    # Convolutional
    pool = 2    # Max-pooling

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_prime(x):
    y = logistic(x)
    return y * (1.0 - y) 

def pseudo_logistic(x):
    return 1.0 / (1.0 + 2 ** -x)

def pseudo_logistic_prime(z):
    y = pseudo_logistic(z)
    return mt.log(2) * y * (1.0 - y) 

def tanh(x):
    y = np.exp(-2 * x)
    return  (1.0 - y) / (1.0 + y)

def tanh_prime(x):
    y = tanh(x)
    return 1.0 - y * y

def rectlin(x):
    xc = x.copy()
    xc[xc < 0] = 0
    return xc

def rectlin_prime(x):
    xc = x.copy()
    xc[xc < 0] = 0
    xc[xc != 0] = 1
    return xc

def get_prime(func):
    if func == logistic:
        return logistic_prime
    if func == pseudo_logistic:
        return pseudo_logistic_prime
    if func == tanh:
        return tanh_prime
    if func == rectlin:
        return rectlin_prime

def get_loss_de(func):
    if func == ce:
        return ce_de
    if func == sse:
        return sse_de

def ce(outputs, targets):
    return np.mean(-targets * np.log(outputs) - \
                   (1 - targets) * np.log(1 - outputs))

def ce_de(outputs, targets):
    return (outputs - targets) / (outputs * (1.0 - outputs)) 

def sse(outputs, targets):
    """ Sum of squared errors """
    return 0.5 * np.sum((outputs - targets) ** 2)

def sse_de(outputs, targets):
    """ Derivative of SSE with respect to the output """
    return (outputs - targets)

def init_weights(shape):
    return np.random.uniform(-0.1, 0.1, shape)

def error_rate(preds, labels):
    return 100.0 * np.mean(np.not_equal(preds, labels))

def append_bias(data):
    """ Append a column of ones. """
    return np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)

class Layer:
    def __init__(self, nin, nout, g):
        self.weights = init_weights((nin, nout))
        self.g = g
        self.gprime = get_prime(g)
        self.nout = nout
        
    def fprop(self, inputs):
        inputs = append_bias(inputs)
        self.z = np.dot(inputs, self.weights)
        self.y = self.g(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.gprime(self.z)

    def update(self, inputs, epsilon):
        inputs = append_bias(inputs)
        self.weights -= epsilon * np.dot(inputs.T, self.delta)

    def error(self):
        return np.dot(self.delta, self.weights[:-1, :].T)

