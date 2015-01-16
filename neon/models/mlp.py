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
from neon.util.param import opt_param, req_param

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
        req_param(self, ['layers', 'batch_size'])
        self.nlayers = len(self.layers)
        self.result = 0
        self.cost.initialize(kwargs)
        assert self.layers[-1].nout <= 2 ** 15

    def fit(self, dataset):
        """
        Learn model weights on the given dataset.
        """
        if self.dist_mode == 'datapar':
            valid_batch_size = (self.batch_size != dataset.batch_size /
                                dataset.num_procs)
            if valid_batch_size:
                raise ValueError('Dataset batch size must be Model batch '
                                 'size * num_procs. Model batch size of %d '
                                 'might work.' % (dataset.batch_size /
                                                  dataset.num_procs))

        for layer in self.layers:
            logger.info("%s", str(layer))
        ds = dataset
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
            num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
            ds.preprocess_done = False
            ds.init_mini_batch_producer(batch_size=self.batch_size,
                                        batch_type='training')
        assert 'batch_size' in self.__dict__
        logger.info('commencing model fitting')
        # force preprocess even if done earlier by setting to False
        error = self.backend.empty((1, 1))
        for epoch in range(self.num_epochs):
            error.fill(0)
            for batch in range(num_batches):
                if ds.macro_batched:
                    # load mini-batch for macro_batched dataset
                    inputs, targets = ds.get_mini_batch()
                    self.fprop(inputs)
                    self.bprop(targets, inputs)
                    self.backend.add(error, self.get_error(targets, inputs) /
                                     self.batch_size, error)
                    rem = (batch + 1) % ds.num_minibatches_in_macro
                    if rem == 0:
                        quot = (batch + 1) / ds.num_minibatches_in_macro - 1
                        logger.info("%d.%d logloss= %0.5f",
                                    epoch, quot,
                                    error / (batch + 1.))
                else:
                    inputs_batch = ds.get_batch(inputs, batch)
                    targets_batch = ds.get_batch(targets, batch)
                    self.fprop(inputs_batch)
                    self.bprop(targets_batch, inputs_batch)
                    batch_err = self.get_error(targets_batch, inputs_batch)
                    self.backend.divide(batch_err, self.batch_size, batch_err)
                    self.backend.add(error, batch_err, error)
                self.update(epoch)
            if self.dist_mode == 'datapar':
                error[0, 0] = MPI.COMM_WORLD.reduce(error.asnumpyarray(),
                                                    op=MPI.SUM)
                if MPI.COMM_WORLD.rank == 0:
                    logger.info('epoch: %d, total training error: %0.5f',
                                epoch,
                                error.asnumpyarray() / num_batches /
                                MPI.COMM_WORLD.size)
            else:
                logger.info('epoch: %d, total training error: %0.5f', epoch,
                            error.asnumpyarray() / num_batches)
            for layer in self.layers:
                logger.debug("%s", layer)
        if ds.macro_batched:
            ds.del_mini_batch_producer()

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

    def predict(self, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given dataset.
        """
        ds = self.dataset
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

        return preds

    def get_error(self, targets, inputs):
        return self.cost.apply_function(targets)

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers:
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs):  # , inputs2, targets2):
        i = self.nlayers - 1
        error = self.cost.apply_derivative(targets)
        batch_size = self.batch_size
        if self.dist_mode == 'datapar':
            batch_size *= MPI.COMM_WORLD.size
        self.backend.divide(error, batch_size, out=error)

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
        sums = self.backend.empty((1, self.batch_size))
        batch_sum = self.backend.empty((1, 1))
        result = self.backend.zeros((1, 1))
        for batch in range(num_batches):
            self.backend.clip(preds[batch], eps, 1.0 - eps, out=preds[batch])
            sums = self.backend.sum(preds[batch], axes=0, out=sums)
            # XXX: work around lack of broadcasting in gpu backend.
            for row in range(preds[batch].shape[0]):
                temp[row] = sums

            self.backend.divide(preds[batch], temp, temp)
            self.backend.log(temp, out=temp)
            self.backend.multiply(targets[batch], temp, temp)
            self.backend.sum(temp, axes=None, out=batch_sum)
            self.backend.add(result, batch_sum, result)
        self.backend.multiply(result, -1, result)
        return self.backend.divide(result, self.batch_size * num_batches,
                                   result)

    def misclass_rate(self, preds, targets):
        # Simple misclassification error.
        num_batches = len(preds)
        labels = self.backend.empty((1, self.batch_size))
        predlabels = self.backend.empty((1, self.batch_size))
        misclass = self.backend.empty((1, self.batch_size))
        batch_sum = self.backend.empty((1, 1))
        misclass_sum = self.backend.zeros((1, 1))
        for batch in range(num_batches):
            self.backend.argmax(targets[batch], axis=0, out=labels)
            self.backend.argmax(preds[batch], axis=0, out=predlabels)
            self.backend.not_equal(predlabels, labels, misclass)
            self.backend.sum(misclass, axes=None, out=batch_sum)
            self.backend.add(misclass_sum, batch_sum, misclass_sum)
        return self.backend.divide(misclass_sum,
                                   num_batches * self.batch_size, misclass_sum)

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
        ds = datasets
        preds = predictions
        targets = ds.get_targets(train=True, test=True, validation=True)
        for item in items:
            if item not in targets:
                continue
            if item not in preds:
                continue
            self.result = self.misclass_rate(preds[item], targets[item])
            logloss = self.logloss(preds[item], targets[item])
            logging.info("%s set misclass rate: %0.5f%% logloss %0.5f",
                         item, 100 * self.result.asnumpyarray(),
                         logloss.asnumpyarray())
        # TODO: return values instead?

    def predict_and_error(self, dataset):
        for layer in self.layers:
            layer.set_train_mode(False)
        be = self.backend
        preds = be.empty((1, self.batch_size))
        labels = be.empty((1, self.batch_size))
        batch_err = be.empty((1, 1))
        tot_err = be.empty((1, 1))
        for setname in ['train', 'test', 'validation']:
            if dataset.has_set(setname) is False:
                continue
            num_batches = dataset.init_mini_batch_producer(
                batch_size=self.batch_size, setname=setname, predict=True)
            nrecs = self.batch_size * num_batches
            preds = be.empty((1, self.batch_size))
            tot_err.fill(0)
            for batch in range(num_batches):
                inputs, targets = dataset.get_mini_batch(batch)
                self.fprop(inputs)
                be.argmax(self.get_classifier_output(),
                                       axis=0,
                                       out=preds)
                be.argmax(targets, axis=0, out=labels)
                be.not_equal(labels, preds, preds)
                be.sum(preds, axes=None, out=batch_err)
                be.add(tot_err, batch_err, tot_err)
            logging.info("%s set misclass rate: %0.5f%%" % (
                setname, 100 * tot_err.asnumpyarray() / nrecs))
            self.result = tot_err.asnumpyarray()[0][0] / nrecs
            dataset.del_mini_batch_producer()

    def get_classifier_output(self):
        return self.layers[-1].output


class MLPB(MLP):

    """
    Fully connected, feed-forward, multi-layer perceptron model
    """

    def __init__(self, **kwargs):
        self.dist_mode = None
        self.__dict__.update(kwargs)
        req_param(self, ['layers', 'batch_size'])
        opt_param(self, ['step_print'], -1)
        opt_param(self, ['accumulate'], False)
        self.result = 0
        kwargs = {"backend": self.backend, "batch_size": self.batch_size,
                  "accumulate": self.accumulate}
        self.data_layer = self.layers[0]
        self.cost_layer = self.layers[-1]
        self.class_layer = self.layers[-2]

        self.link_and_initialize(self.layers, kwargs)

        assert self.layers[-1].nout <= 2 ** 15

    def link_and_initialize(self, layer_list, kwargs, initlayer=None):
        for ll, pl in zip(layer_list, [initlayer] + layer_list[:-1]):
            ll.set_previous_layer(pl)
            ll.initialize(kwargs)

    def fprop(self):
        for ll, pl in zip(self.layers, [None] + self.layers[:-1]):
            y = None if pl is None else pl.output
            ll.fprop(y)

    def bprop(self):
        for ll, nl in zip(reversed(self.layers),
                          reversed(self.layers[1:] + [None])):
            error = None if nl is None else nl.berror
            ll.bprop(error)

    def print_layers(self, debug=False):
        printfunc = logger.debug if debug else logger.info
        for layer in self.layers:
            printfunc("%s", str(layer))

    def get_classifier_output(self):
        return self.class_layer.output

    def fit(self, dataset):
        """
        Learn model weights on the given datasets.
        """
        error = self.backend.empty((1, 1))
        mb_id = self.backend.empty((1, 1))
        self.print_layers()
        self.data_layer.init_dataset(dataset)
        self.data_layer.use_set('train')
        logger.info('commencing model fitting')
        for epoch in range(self.num_epochs):
            error.fill(0.0)
            mb_id.fill(1)
            self.data_layer.reset_counter()
            while self.data_layer.has_more_data():
                self.fprop()
                self.bprop()
                self.update(epoch)
                self.backend.add(error, self.cost_layer.get_cost(), error)
                if self.step_print > 0 and mb_id % self.step_print:
                    logger.info('%d.%d logloss=%0.5f', epoch,
                                mb_id / self.step_print - 1, error / mb_id)
                self.backend.sum(mb_id, 1, mb_id)
            logger.info('epoch: %d, total training error: %0.5f', epoch,
                        error.asnumpyarray() / self.data_layer.num_batches)
            self.print_layers(debug=True)
        self.data_layer.cleanup()

    def predict_and_error(self, dataset=None):
        if dataset is not None:
            self.data_layer.init_dataset(dataset)
        predlabels = self.backend.empty((1, self.batch_size))
        labels = self.backend.empty((1, self.batch_size))
        misclass = self.backend.empty((1, self.batch_size))
        logloss_sum = self.backend.empty((1, 1))
        misclass_sum = self.backend.empty((1, 1))
        batch_sum = self.backend.empty((1, 1))
        for setname in ['train', 'test', 'validation']:
            if self.data_layer.has_set(setname) is False:
                continue
            self.data_layer.use_set(setname, predict=True)
            self.data_layer.reset_counter()
            misclass_sum.fill(0.0)
            logloss_sum.fill(0.0)
            nrecs = self.batch_size * self.data_layer.num_batches
            while self.data_layer.has_more_data():
                self.fprop()
                probs = self.get_classifier_output()
                targets = self.data_layer.targets
                self.backend.argmax(targets, axis=0, out=labels)
                self.backend.argmax(probs, axis=0, out=predlabels)
                self.backend.not_equal(predlabels, labels, misclass)
                self.backend.sum(misclass, axes=None, out=batch_sum)
                self.backend.add(misclass_sum, batch_sum, misclass_sum)
                self.backend.sum(self.cost_layer.cost.apply_logloss(targets),
                                 axes=None, out=batch_sum)
                self.backend.add(logloss_sum, batch_sum, logloss_sum)
            logging.info("%s set misclass rate: %0.5f%% logloss %0.5f" % (
                setname, 100 * misclass_sum.asnumpyarray() / nrecs,
                logloss_sum.asnumpyarray() / nrecs))
            self.result = misclass_sum.asnumpyarray()[0, 0] / nrecs
            self.data_layer.cleanup()
