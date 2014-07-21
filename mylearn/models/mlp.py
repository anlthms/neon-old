"""
Simple multi-layer perceptron model.
"""

import logging
import math

from mylearn.models.layer import Layer
from mylearn.models.model import Model


logger = logging.getLogger(__name__)


class MLP(Model):
    """
    Adaptation of Anil's reference mlp7 multi-layer perceptron.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing model fitting')
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
        nrecs, nin = inputs.shape
        backend = datasets[0].backend
        self.loss_fn = getattr(backend, self.loss_fn)
        self.de = backend.get_derivative(self.loss_fn)
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        layers = []
        for i in xrange(self.nlayers):
            layer = self.lcreate(backend, nin, self.layers[i])
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
                self.fprop(inputs[start_idx:end_idx])
                self.bprop(targets[start_idx:end_idx])
                self.update(inputs[start_idx:end_idx], self.learning_rate,
                            epoch, self.momentum)
                error += self.loss_fn(self.layers[-1].y,
                                      targets[start_idx:end_idx])
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))
            # for layer in self.layers:
            #    logger.info('layer:\n\t%s' % str(layer))

    def predict(self, datasets, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given datasets.
        """
        res = []
        for dataset in datasets:
            inputs = dataset.get_inputs(train, test, validation)
            preds = dict()
            if train and 'train' in inputs:
                outputs = self.fprop(inputs['train'])
                preds['train'] = dataset.backend.argmax(outputs, axis=1)
            if test and 'test' in inputs:
                outputs = self.fprop(inputs['test'])
                preds['test'] = dataset.backend.argmax(outputs, axis=1)
            if validation and 'validation' in inputs:
                outputs = self.fprop(inputs['validation'])
                preds['validation'] = dataset.backend.argmax(outputs, axis=1)
            if len(preds) is 0:
                logger.error("must specify >=1 of: train, test, validation")
            res.append(preds)
        return res

    def lcreate(self, backend, nin, conf):
        if conf['connectivity'] == 'full':
            # Add 1 for the bias input.
            return Layer(conf['name'], backend, nin + 1,
                         nout=conf['num_nodes'],
                         act_fn=conf['activation_fn'],
                         weight_init=conf['weight_init'])

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers:
            y = layer.fprop(y)
        return y

    def bprop(self, targets):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        lastlayer.bprop(self.de(lastlayer.y, targets) / targets.shape[0])
        while i > 0:
            error = self.layers[i].error()
            i -= 1
            self.layers[i].bprop(error)

    def update(self, inputs, epsilon, epoch, momentum):
        self.layers[0].update(inputs, epsilon, epoch, momentum)
        for i in xrange(1, self.nlayers):
            self.layers[i].update(self.layers[i - 1].y, epsilon, epoch,
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
                                                    ds.backend.nonzero(
                                                    targets[item]))
                    err = ds.backend.mean(misclass)
                    logging.info("%s set misclass rate: %0.5f%%" % (
                        item, 100 * err))
        # TODO: return values instead?
