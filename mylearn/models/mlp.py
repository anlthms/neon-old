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
            layer = self.lcreate(self.backend, nin, self.layers[i])
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
                self.fprop(inputs.take(range(start_idx, end_idx), axis=0))
                self.bprop(targets.take(range(start_idx, end_idx), axis=0))
                self.update(inputs.take(range(start_idx, end_idx), axis=0),
                            self.learning_rate, epoch, self.momentum)
                error += self.cost.apply_function(self.layers[-1].output,
                                                  targets.take(range(start_idx,
                                                                     end_idx),
                                                               axis=0))
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))

    def predict_set(self, inputs):
        nrecs = inputs.shape[0]
        outputs = self.backend.zeros((nrecs, self.layers[-1].nout))
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for batch in xrange(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, nrecs)
            self.fprop(inputs.take(range(start_idx, end_idx), axis=0))
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

    def lcreate(self, backend, nin, conf):
        if conf['connectivity'] == 'full':
            # instantiate the activation function class from string name given
            activation = Factory.create(type=conf['activation'])
            # Add 1 for the bias input.
            return Layer(conf['name'], backend, nin + 1,
                         nout=conf['num_nodes'],
                         activation=activation,
                         weight_init=conf['weight_init'])

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers:
            layer.fprop(y)
            y = layer.output
        return y

    def bprop(self, targets):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        lastlayer.bprop(self.cost.apply_derivative(lastlayer.output, targets) /
                        targets.shape[0])
        while i > 0:
            error = self.layers[i].error()
            i -= 1
            self.layers[i].bprop(error)

    def update(self, inputs, epsilon, epoch, momentum):
        self.layers[0].update(inputs, epsilon, epoch, momentum)
        for i in xrange(1, self.nlayers):
            self.layers[i].update(self.layers[i - 1].output, epsilon, epoch,
                                  momentum)

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