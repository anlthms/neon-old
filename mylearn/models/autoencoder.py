"""
Contains code to train stacked autoencoder models and run inference.
"""

import logging
import math

from mylearn.models.layer import AELayer
from mylearn.models.mlp import MLP
from mylearn.util.factory import Factory

logger = logging.getLogger(__name__)


class Autoencoder(MLP):
    """
    Adaptation of multi-layer perceptron.
    """

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing model fitting')
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_inputs(train=True)['train']
        nrecs, nin = inputs.shape
        self.backend = datasets[0].backend
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        tempbuf = self.backend.zeros((self.batch_size, targets.shape[1]))
        self.temp = [tempbuf, tempbuf.copy()]
        layers = []
        assert self.nlayers % 2 == 0
        for i in xrange(self.nlayers):
            if i >= self.nlayers / 2:
                weights = layers[self.nlayers - i - 1].weights.transpose()
            else:
                weights = None
            layer = self.lcreate(self.backend, nin, self.layers[i], i, weights)
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
                preds['train'] = self.predict_set(inputs['train'])
            if test and 'test' in inputs:
                preds['test'] = self.predict_set(inputs['test'])
            if validation and 'validation' in inputs:
                preds['validation'] = self.predict_set(inputs['validation'])
            if len(preds) is 0:
                logger.error("must specify >=1 of: train, test, validation")
            res.append(preds)
        return res

    def lcreate(self, backend, nin, conf, pos, weights):
        if conf['connectivity'] == 'full':
            # instantiate the activation function class from string name given
            activation = Factory.create(type=conf['activation'])
            return AELayer(conf['name'], backend, self.batch_size, pos,
                           self.learning_rate, nin,
                           nout=conf['num_nodes'],
                           activation=activation,
                           weight_init=conf['weight_init'],
                           weights=weights)

    # TODO: move out to separate config params and module.
    def error_metrics(self, datasets, predictions, train=True, test=True,
                      validation=True):
        # Reconstruction error.
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
            targets = ds.get_inputs(train=True, test=True, validation=True)
            for item in items:
                if item in targets and item in preds:
                    tempbuf = self.backend.zeros((preds[item].shape[0],
                                                  preds[item].shape[1]))
                    temp = [tempbuf, tempbuf.copy()] 
                    err = self.cost.apply_function(self.backend,
                                                   preds[item],
                                                   targets[item],
                                                   temp)
                    logging.info("%s set reconstruction error : %0.5f" %
                                 (item, err))
