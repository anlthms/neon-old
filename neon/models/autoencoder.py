# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train stacked autoencoder models and run inference.
"""

import logging

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
        ds = datasets[0]
        inputs = ds.get_inputs(train=True)['train']
        targets = ds.get_inputs(train=True)['train']
        self.nlayers = len(self.layers)
        assert 'batch_size' in self.__dict__
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        tempbuf = self.backend.empty((self.layers[-1].nout, self.batch_size),
                                     self.temp_dtype)
        self.temp = [tempbuf, tempbuf.copy()]

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(inputs.nbatches):
                inputs_batch = ds.get_batch(inputs, batch)
                targets_batch = ds.get_batch(targets, batch)
                self.fprop(inputs_batch)
                self.bprop(targets_batch, inputs_batch)
                error += self.cost.apply_function(
                    self.backend, self.layers[-1].output,
                    targets_batch,
                    self.temp)
                self.update(epoch)
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / inputs.nbatches))

    def predict(self, datasets, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given datasets.
        """
        res = []
        for dataset in datasets:
            inputs = dataset.get_inputs(train, test, validation)
            preds = dict()
            if train and 'train' in inputs:
                preds['train'] = self.predict_set(dataset, inputs['train'])
            if test and 'test' in inputs:
                preds['test'] = self.predict_set(dataset, inputs['test'])
            if validation and 'validation' in inputs:
                preds['validation'] = self.predict_set(dataset,
                                                       inputs['validation'])
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
