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
        for req_param in ['layers', 'batch_size']:
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
            num_batches = len(inputs)
        else:
            if ds.start_train_batch == -1:
                nrecs = ds.max_file_index
            else:
                nrecs = ds.output_batch_size * \
                    (ds.end_train_batch - ds.start_train_batch + 1)
            ds.cur_train_macro_batch = ds.start_train_batch
            num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))

        logger.info('commencing model fitting')
        for epoch in range(self.num_epochs):
            error = 0.0
            for batch in range(num_batches):
                if ds.macro_batched:
                    # load mini-batch for macro_batched dataset
                    logger.info('loading mb %d', batch)
                    inputs, targets = ds.get_mini_batch(
                        self.batch_size, 'training')
                    logger.info('done loading mb %d', batch)
                    self.fprop(inputs)
                    self.bprop(targets, inputs)
                    error += self.get_error(targets, inputs) / self.batch_size
                else:
                    inputs_batch = ds.get_batch(inputs, batch)
                    targets_batch = ds.get_batch(targets, batch)
                    self.fprop(inputs_batch)
                    self.bprop(targets_batch, inputs_batch)
                    error += self.get_error(
                        targets_batch, inputs_batch) / self.batch_size
                self.update(epoch)
            if self.dist_mode == 'datapar':
                error = MPI.COMM_WORLD.reduce(error, op=MPI.SUM)
                if MPI.COMM_WORLD.rank == 0:
                    logger.info('epoch: %d, total training error: %0.5f',
                                epoch,
                                error / num_batches / MPI.COMM_WORLD.size)
            else:
                logger.info('epoch: %d, total training error: %0.5f', epoch,
                            error / num_batches)
            for layer in self.layers:
                logger.debug("%s", layer)

    def predict_set(self, ds, inputs):
        for layer in self.layers:
            layer.set_train_mode(False)
        num_batches = len(inputs)
        nout = self.layers[-1].nout
        preds = []
        for batch in range(num_batches):
            inputs_batch = ds.get_batch(inputs, batch)
            preds_batch = self.backend.empty((nout, self.batch_size))
            self.fprop(inputs_batch)
            preds_batch[:] = self.get_classifier_output()
            preds.append(preds_batch)
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

    def get_error(self, targets, inputs):
        return self.cost.apply_function(targets)

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

    def logloss(self, preds, targets, eps=1e-15):
        num_batches = len(preds)
        temp = self.backend.empty(preds[0].shape)
        result = 0.
        for batch in range(num_batches):
            self.backend.clip(preds[batch], eps, 1.0-eps, out=temp)
            self.backend.log(temp, out=temp)
            self.backend.multiply(targets[batch], temp, temp)
            result += self.backend.sum(temp)
        return -result / (self.batch_size * num_batches)

    def misclass_rate(self, preds, targets):
        # Simple misclassification error.
        num_batches = len(preds)
        labels = self.backend.empty((1, self.batch_size))
        predlabels = self.backend.empty((1, self.batch_size))
        misclass = self.backend.empty((1, self.batch_size))
        misclass_sum = 0
        for batch in range(num_batches):
            self.backend.argmax(targets[batch], axis=0, out=labels)
            self.backend.argmax(preds[batch], axis=0, out=predlabels)
            self.backend.not_equal(predlabels, labels, misclass)
            misclass_sum += self.backend.sum(misclass)
        return misclass_sum / (num_batches * self.batch_size)

    # TODO: move out to separate config params and module.
    def error_metrics(self, datasets, predictions, train=True, test=True,
                      validation=True):
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
                if item not in targets:
                    continue
                if item not in preds:
                    continue
                num_batches = len(preds[item])
                self.result = self.misclass_rate(
                    ds, num_batches, preds[item], targets[item])
                logloss = self.logloss(
                    ds, num_batches, preds[item], targets[item])
                logging.info("%s set misclass rate: %0.5f%% logloss %0.5f",
                             item, 100 * self.result, logloss)
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
                dataset.backend.argmax(self.get_classifier_output(),
                                       axis=0,
                                       out=preds)
                dataset.backend.not_equal(targets, preds, preds)
                err += dataset.backend.sum(preds)
            logging.info("%s set misclass rate: %0.5f%%" % (
                batch_type, 100 * err / nrecs))

    def get_classifier_output(self):
        return self.layers[-1].output


class MLPB(MLP):

    """
    Fully connected, feed-forward, multi-layer perceptron model
    """

    def __init__(self, **kwargs):
        self.dist_mode = None
        self.__dict__.update(kwargs)
        for req_param in ['layers', 'batch_size']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)
        self.nlayers = len(self.layers)
        self.result = 0
        kwargs = {"backend": self.backend, "batch_size": self.batch_size}
        self.data_layer = self.layers[0]
        self.cost_layer = self.layers[-1]
        self.class_layer = self.layers[-2]

        for ll, pl in zip(self.layers, [None] + self.layers[:-1]):
            ll.set_previous_layer(pl)
            ll.initialize(kwargs)

        assert self.layers[-1].nout <= 2 ** 15

    def fprop(self):
        for ll, pl in zip(self.layers, [None] + self.layers[:-1]):
            y = None if pl is None else pl.output
            ll.fprop(y)

    def bprop(self):
        for ll, nl in zip(reversed(self.layers),
                          reversed(self.layers[1:] + [None])):
            error = None if nl is None else nl.berror
            ll.bprop(error)

    def predict(self, items=['train', 'test', 'validation']):
        """
        Generate and return predictions on the given datasets.
        """
        for layer in self.layers:
            layer.set_train_mode(False)
        preds = dict()
        for sn in items:
            res = self.predict_set(sn)
            if res is not None:
                preds[sn] = res
        return preds

    def print_layers(self, debug=False):
        printfunc = logger.debug if debug else logger.info
        for layer in self.layers:
            printfunc("%s", str(layer))

    def error_metrics(self, ds, preds, items=['train', 'test', 'validation']):
        targets = ds.get_targets(train=True, test=True, validation=True)
        for item in items:
            if item not in targets or item not in preds:
                continue
            self.result = self.misclass_rate(preds[item], targets[item])
            logloss = self.logloss(preds[item], targets[item])
            logging.info("%s set misclass rate: %0.5f%% logloss %0.5f",
                         item, 100 * self.result, logloss)

    def predict_set(self, setname):
        if not self.data_layer.has_set(setname):
            return None
        self.data_layer.use_set(setname)
        self.data_layer.reset_counter()
        preds = []
        while self.data_layer.has_more_data():
            self.fprop()
            preds.append(self.get_classifier_output().copy())
        return preds

    def get_classifier_output(self):
        return self.class_layer.output

    def fit(self, dataset):
        """
        Learn model weights on the given datasets.
        """
        self.print_layers()
        self.data_layer.use_set('train')
        logger.info('commencing model fitting')
        for epoch in range(self.num_epochs):
            error = 0.0
            self.data_layer.reset_counter()
            while self.data_layer.has_more_data():
                self.fprop()
                self.bprop()
                self.update(epoch)
                error += self.cost_layer.get_cost()
            logger.info('epoch: %d, total training error: %0.5f', epoch,
                        error / self.data_layer.num_batches)
            self.print_layers(debug=True)
