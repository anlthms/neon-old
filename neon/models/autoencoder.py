# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train stacked autoencoder models and run inference.
"""

import logging
import math

from neon.models.mlp import MLP

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
        nrecs = inputs.shape[inputs.major_axis()]
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        tempbuf = self.backend.alloc(self.batch_size,
                                     targets.shape[targets.minor_axis()])
        self.temp = [tempbuf, tempbuf.copy()]

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.fprop(inputs.get_minor_slice(start_idx, end_idx))
                self.bprop(targets.get_minor_slice(start_idx, end_idx),
                           inputs.get_minor_slice(start_idx, end_idx),
                           epoch)
                error += self.cost.apply_function(
                    self.backend, self.layers[-1].output,
                    targets.get_minor_slice(start_idx, end_idx),
                    self.temp)
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))

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
                    tempbuf = self.backend.zeros((preds[item].shape))
                    temp = [tempbuf, tempbuf.copy()]
                    err = self.cost.apply_function(self.backend,
                                                   preds[item],
                                                   targets[item],
                                                   temp)
                    logging.info("%s set reconstruction error : %0.5f" %
                                 (item, err))
