"""
Simple multi-layer perceptron model.
"""

import logging
import math

from mylearn.models.layer import Layer
from mylearn.models.model import Model
from mylearn.util.factory import Factory


logger = logging.getLogger(__name__)


class MLP(Model):
    """
    Fully connected, feed-forward, multi-layer perceptron model
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if isinstance(self.cost, str):
            self.cost = Factory.create(type=self.cost)

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing model fitting')
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
        nrecs, nin = inputs.shape
        self.backend = datasets[0].backend
        self.backend.rng_init()
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        layers = []
        for i in xrange(self.nlayers):
            layer = self.lcreate(self.backend, nin, self.layers[i], i)
            logger.info('created layer:\n\t%s' % str(layer))
            layers.append(layer)
            nin = layer.nout
        self.layers = layers
        tempbuf = self.backend.zeros((self.batch_size, targets.shape[1]))
        self.temp = [tempbuf, tempbuf.copy()]

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.fprop(inputs[start_idx:end_idx])
                self.bprop(targets[start_idx:end_idx],
                           inputs[start_idx:end_idx],
                           epoch, self.momentum)
                error += self.cost.apply_function(self.backend,
                                                  self.layers[-1].output,
                                                  targets[start_idx:end_idx],
                                                  self.temp)
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))

    def predict_set(self, inputs):
        nrecs = inputs.shape[0]
        outputs = self.backend.zeros((nrecs, self.layers[-1].nout))
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for batch in xrange(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, nrecs)
            self.fprop(inputs[start_idx:end_idx])
            outputs[start_idx:end_idx, :] = self.layers[-1].output
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
                outputs = self.predict_set(inputs['train'])
                preds['train'] = dataset.backend.argmax(outputs, axis=1)
            if test and 'test' in inputs:
                outputs = self.predict_set(inputs['test'])
                preds['test'] = dataset.backend.argmax(outputs, axis=1)
            if validation and 'validation' in inputs:
                outputs = self.predict_set(inputs['validation'])
                preds['validation'] = dataset.backend.argmax(outputs, axis=1)
            if len(preds) is 0:
                logger.error("must specify >=1 of: train, test, validation")
            res.append(preds)
        return res

    def lcreate(self, backend, nin, conf, pos):
        if conf['connectivity'] == 'full':
            # instantiate the activation function class from string name given
            activation = Factory.create(type=conf['activation'])
            # Add 1 for the bias input.
            return Layer(conf['name'], backend, self.batch_size, pos,
                         self.learning_rate,
                         nin + 1,
                         nout=conf['num_nodes'],
                         activation=activation,
                         weight_init=conf['weight_init'])

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers:
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs, epoch, momentum):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        error = self.cost.apply_derivative(self.backend,
                                           lastlayer.output, targets,
                                           self.temp)
        self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                            out=error)
        # Update the output layer.
        lastlayer.bprop(error, self.layers[i - 1].output, epoch, momentum)
        while i > 1:
            i -= 1
            self.layers[i].bprop(self.layers[i + 1].berror,
                                 self.layers[i - 1].output,
                                 epoch, momentum)
        # Update the first hidden layer.
        self.layers[i - 1].bprop(self.layers[i].berror, inputs, epoch, momentum)

    # TODO: move out to separate config params and module.
    def error_metrics(self, datasets, predictions, train=True, test=True,
                      validation=True):
        # simple misclassification error
        items = []
        if train:
            items.append('train')
        if test:
            items.append('test')
        if validation:
            items.append('validation')
        for idx in xrange(len(datasets)):
            ds = datasets[idx]
            preds = predictions[idx]
            targets = ds.get_targets(train=True, test=True, validation=True)
            for item in items:
                if item in targets and item in preds:
                    misclass = ds.backend.not_equal(preds[item],
                                                    ds.backend.argmax(
                                                    targets[item], axis=1))
                    err = ds.backend.mean(misclass)
                    logging.info("%s set misclass rate: %0.5f%%" % (
                        item, 100 * err))
        # TODO: return values instead?
