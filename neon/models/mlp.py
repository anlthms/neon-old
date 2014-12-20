# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple multi-layer perceptron model.
"""

import logging
import math

from neon.models.model import Model
from neon.util.compat import MPI_INSTALLED, range

if MPI_INSTALLED:
    from mpi4py import MPI

logger = logging.getLogger(__name__)


class MLP(Model):

    """
    Fully connected, feed-forward, multi-layer perceptron model
    """

    def __init__(self, **kwargs):
        self.dist_mode = None
        self.__dict__.update(kwargs)
        for req_param in ['layers']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)
        self.nlayers = len(self.layers)
        self.result = 0
        assert self.layers[-1].nout <= 2 ** 15

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        if self.dist_mode == 'datapar':
            valid_batch_size = (self.batch_size != datasets[0].batch_size /
                                datasets[0].num_procs)
            if valid_batch_size:
                raise ValueError('Dataset batch size must be Model batch '
                                 'size * num_procs. Model batch size of %d '
                                 'might work.' % (datasets[0].batch_size /
                                                  datasets[0].num_procs))

        for layer in self.layers:
            logger.info("%s", str(layer))
        ds = datasets[0]
        if not ds.macro_batched:
            inputs = ds.get_inputs(train=True)['train']
            targets = ds.get_targets(train=True)['train']
            num_batches = inputs.nbatches
        else:
            if ds.start_train_batch == -1:
                nrecs = ds.max_file_index
            else:
                nrecs = ds.output_batch_size * \
                    (ds.end_train_batch - ds.start_train_batch + 1)
            ds.cur_train_macro_batch = ds.start_train_batch
            num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))

        assert 'batch_size' in self.__dict__
        logger.info('commencing model fitting')
        for epoch in range(self.num_epochs):
            error = 0.0
            for batch in range(num_batches):  # inputs.nbatches
                if ds.macro_batched:
                    # load mini-batch for macro_batched dataset
                    logger.info('loading mb %d', batch)
                    inputs, targets = ds.get_mini_batch(
                        self.batch_size, 'training')
                    logger.info('done loading mb %d', batch)
                    self.fprop(inputs)
                    self.bprop(targets, inputs)
                    error += self.cost.apply_function(targets)
                else:
                    inputs_batch = ds.get_batch(inputs, batch)
                    targets_batch = ds.get_batch(targets, batch)
                    self.fprop(inputs_batch)
                    self.bprop(targets_batch, inputs_batch)
                    error += self.cost.apply_function(targets_batch)
                self.update(epoch)
            if self.dist_mode == 'datapar':
                error = MPI.COMM_WORLD.reduce(error, op=MPI.SUM)
                if MPI.COMM_WORLD.rank == 0:
                    logger.info('epoch: %d, total training error: %0.5f',
                                epoch,
                                error / inputs.nbatches / MPI.COMM_WORLD.size)
            else:
                logger.info('epoch: %d, total training error: %0.5f', epoch,
                            error / num_batches)
            for layer in self.layers:
                logger.debug("%s", layer)

    def predict_set(self, ds, inputs):
        nout = self.layers[-1].nout
        preds = self.backend.empty((inputs.nbatches * nout, self.batch_size))
        preds.nrows = nout

        for layer in self.layers:
            layer.set_train_mode(False)

        for batch in range(inputs.nbatches):
            inputs_batch = ds.get_batch(inputs, batch)
            preds_batch = ds.get_batch(preds, batch)
            self.fprop(inputs_batch)
            preds_batch[:] = self.layers[-1].output
        return preds

    def predict(self, datasets, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given datasets.
        """
        res = []

        for ds in datasets:
            inputs = ds.get_inputs(train=train, test=test,
                                   validation=validation)
            preds = dict()
            if train and 'train' in inputs:
                preds['train'] = self.predict_set(ds, inputs['train'])
            if test and 'test' in inputs:
                preds['test'] = self.predict_set(ds, inputs['test'])
            if validation and 'validation' in inputs:
                preds['validation'] = self.predict_set(ds,
                                                       inputs['validation'])
            if len(preds) is 0:
                logger.error("must specify >=1 of: train, test, validation")
            res.append(preds)
        return res

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers:
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs):
        i = self.nlayers - 1
        error = self.cost.apply_derivative(targets)
        batch_size = self.batch_size
        if self.dist_mode == 'datapar':
            batch_size *= MPI.COMM_WORLD.size
        self.backend.divide(error, self.backend.wrap(batch_size), out=error)

        while i > 0:
            self.layers[i].bprop(error, self.layers[i - 1].output)
            error = self.layers[i].berror
            i -= 1
        self.layers[i].bprop(error, inputs)

    def update(self, epoch):
        for layer in self.layers:
            layer.update(epoch)

    def logloss(self, ds, preds, targets, eps=1e-15):
        self.backend.clip(preds, eps, 1.0 - eps, out=preds)
        temp = self.backend.empty(preds.shape)
        temp.nrows = preds.nrows

        sums = self.backend.empty((1, self.batch_size))
        for batch in range(targets.nbatches):
            pred_batch = ds.get_batch(preds, batch)
            temp_batch = ds.get_batch(temp, batch)
            sums = self.backend.sum(pred_batch, axis=0, out=sums)

            # XXX: work around lack of broadcasting in gpu backend.
            for row in range(pred_batch.shape[0]):
                temp_batch[row] = sums

            self.backend.divide(pred_batch, temp_batch, temp_batch)

        self.backend.log(temp, out=temp)
        self.backend.multiply(targets, temp, temp)
        return -self.backend.sum(temp) / (self.batch_size * targets.nbatches)

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
        for idx in range(len(datasets)):
            ds = datasets[idx]
            preds = predictions[idx]
            targets = ds.get_targets(train=True, test=True, validation=True)
            for item in items:
                if item in targets and item in preds:
                    labels = self.backend.empty((targets[item].nbatches,
                                                 self.batch_size))
                    predlabels = self.backend.empty(labels.shape)
                    for batch in range(targets[item].nbatches):
                        targets_batch = ds.get_batch(targets[item], batch)
                        self.backend.argmax(targets_batch, axis=0,
                                            out=labels[batch:(batch + 1)])
                        preds_batch = ds.get_batch(preds[item], batch)
                        self.backend.argmax(preds_batch, axis=0,
                                            out=predlabels[batch:(batch + 1)])

                    misclass = ds.backend.empty(labels.shape)
                    ds.backend.not_equal(predlabels, labels, misclass)
                    self.result = ds.backend.mean(misclass)
                    logging.info(
                        "%s set misclass rate: %0.5f%% logloss %0.5f", item,
                        100 * self.result,
                        self.logloss(ds, preds[item], targets[item], 0.001))
        # TODO: return values instead?

    def predict_and_error(self, dataset):

        for batch_type in ['training', 'validation']:
            if batch_type == 'training':
                nrecs = dataset.output_batch_size * \
                    (dataset.end_train_batch - dataset.start_train_batch + 1)
                dataset.cur_train_macro_batch = dataset.start_train_batch
            elif batch_type == 'validation':
                nrecs = dataset.output_batch_size * \
                    (dataset.end_val_batch - dataset.start_val_batch + 1)
                dataset.cur_val_macro_batch = dataset.start_val_batch
            num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))

            preds = dataset.backend.empty((1, self.batch_size))
            err = 0.
            for batch in range(num_batches):
                inputs, targets = dataset.get_mini_batch(
                    self.batch_size, batch_type, raw_targets=True)
                self.fprop(inputs)
                dataset.backend.argmax(self.layers[-1].output,
                                       axis=0,
                                       out=preds)
                dataset.backend.not_equal(targets, preds, preds)
                err += dataset.backend.sum(preds)
            logging.info("%s set misclass rate: %0.5f%%" % (
                batch_type, 100 * err / nrecs))
