"""
Contains code to train stacked autoencoder models and run inference.
"""

import logging
import math
import numpy as np
from model import Model

logger = logging.getLogger(__name__)


class Layer(object):
    """
    Single NNet layer built to handle data from a particular backend
    """
    def __init__(self, name, backend, nin, nout, act_fn, weight_init, weights=None):
        self.name = name
        self.backend = backend
        if weights == None:
            self.weights = self.backend.gen_weights((nout, nin), weight_init)
        else:
            self.weights = weights
        self.act_fn = getattr(self.backend, act_fn)
        self.act_fn_de = self.backend.get_derivative(self.act_fn)
        self.nin = nin
        self.nout = nout
        self.y = None
        self.z = None
        
    def __str__(self):
        return ("Layer %s: %d inputs, %d nodes, %s act_fn, "
                "utilizing %s backend\n\t"
                "y: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "z: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" % 
                (self.name, self.nin, self.nout, self.act_fn.__name__,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.y),
                 self.backend.min(self.y),
                 self.backend.max(self.y),
                 self.backend.mean(self.z),
                 self.backend.min(self.z),
                 self.backend.max(self.z),
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights)))

    def fprop(self, inputs):
        self.z = self.backend.dot(inputs, self.weights.T())
        if self.act_fn == self.backend.noact:
            self.y = self.z
        else:
            self.y = self.act_fn(self.z)
        return self.y

    def bprop(self, error):
        if self.act_fn_de == self.backend.noact_prime:
            self.delta = error
        else:
            self.delta = error * self.act_fn_de(self.z)

    def update(self, inputs, epsilon, epoch):
        self.weights -= epsilon * self.backend.dot(self.delta.T(), inputs)

    def error(self):
        return self.backend.dot(self.delta, self.weights)


class Network(Model):
    """
    Adaptation of multi-layer perceptron.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing model fitting')
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_inputs(train=True)['train']
        nrecs, nin = inputs.shape
        backend = datasets[0].backend
        self.loss_fn = getattr(backend, self.loss_fn)
        self.loss_fn_de = backend.get_derivative(self.loss_fn) 
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        layers = []
        assert self.nlayers % 2 == 0
        for i in xrange(self.nlayers):
            if i >= self.nlayers / 2:
                weights = layers[self.nlayers - i - 1].weights.transpose()
            else:
                weights = None
            layer = self.lcreate(backend, nin, self.layers[i], weights)
            logger.info('created layer:\n\t%s' % str(layer))
            layers.append(layer)
            nin = layer.nout
        self.layers = layers
        
        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs): 
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.fprop(inputs.take(start_idx, end_idx, axis=0))
                self.bprop(targets.take(start_idx, end_idx, axis=0))
                self.update(inputs.take(start_idx, end_idx, axis=0),
                            self.learning_rate, epoch) 
                error += self.loss_fn(self.layers[-1].y,
                                      targets.take(start_idx, end_idx, axis=0))
            logger.info('epoch: %d, total training error: %0.5f' %
                    (epoch, error / num_batches))

    def predict_set(self, inputs):
        nrecs = inputs.shape[0]
        outputs = np.zeros((nrecs, self.layers[-1].nout)) 
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for batch in xrange(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, nrecs)
            self.fprop(inputs.take(start_idx, end_idx, axis=0))
            outputs[start_idx:end_idx] = self.layers[-1].y.raw()
        return outputs

    def predict(self, datasets, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given datasets.
        """
        res = []
        for dataset in datasets:
            inputs = dataset.get_inputs(train, test, validation)
            preds = dict()
            if train and 'train' in inputs:
                preds['train'] = self.predict_set(inputs['train']) 
            if test and 'test' in inputs:
                preds['test'] = self.predict_set(inputs['test']) 
            if validation and 'validation' in inputs:
                preds['validation'] = self.predict_set(inputs['validation']) 
            if len(preds) is 0:
                logger.error("must specify >=1 of: train, test, validation")
            res.append(preds)
        return res

    def lcreate(self, backend, nin, conf, weights):
        if conf['connectivity'] == 'full':
            return Layer(conf['name'], backend, nin,
                         nout=conf['num_nodes'],
                         act_fn=conf['activation_fn'],
                         weight_init=conf['weight_init'],
                         weights=weights)

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers: 
            y = layer.fprop(y)
        return y

    def bprop(self, targets):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        lastlayer.bprop(self.loss_fn_de(lastlayer.y, targets) / targets.shape[0])
        while i > 0:
            error = self.layers[i].error()
            i -= 1 
            self.layers[i].bprop(error)

    def update(self, inputs, epsilon, epoch):
        self.layers[0].update(inputs, epsilon, epoch)
        for i in xrange(1, self.nlayers):
            self.layers[i].update(self.layers[i - 1].y, epsilon, epoch)

    def cross_entropy(self, outputs, targets):
        return np.mean(-targets * np.log(outputs) - \
                (1 - targets) * np.log(1 - outputs))

    def error_metrics(self, datasets, predictions, train=True, test=True,
                      validation=True):
        # Reconstruction error. 
        items = []
        if train: items.append('train')
        if test: items.append('test')
        if validation: items.append('validation')
        for idx in xrange(len(datasets)):
            ds = datasets[idx]
            preds = predictions[idx]
            targets = ds.get_inputs(train=True, test=True, validation=True)
            for item in items:
                if item in targets and item in preds:
                    err = self.cross_entropy(preds[item], targets[item].raw())
                    logging.info("%s set reconstruction error : %0.5f" %
                            (item, err))
