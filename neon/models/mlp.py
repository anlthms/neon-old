# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple multi-layer perceptron model.
"""

import logging
import math

from neon.models.model import Model
from neon.util.compat import MPI_INSTALLED

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
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        if 'ada' not in self.__dict__:
            self.ada = None
        tempbuf = self.backend.alloc(self.batch_size, self.layers[-1].nout,
                                     self.temp_dtype)
        self.temp = [tempbuf, tempbuf.copy()]
        self.result = 0

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
            logger.info("%s" % str(layer))
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
        nrecs = inputs.shape[inputs.major_axis()]
        assert 'batch_size' in self.__dict__

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        logger.info('commencing model fitting')
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
            if self.dist_mode == 'datapar':
                error = MPI.COMM_WORLD.reduce(error, op=MPI.SUM)
                if MPI.COMM_WORLD.rank == 0:
                    logger.info('epoch: %d, total training error: %0.5f' %
                                (epoch, error / num_batches /
                                    MPI.COMM_WORLD.size))
            else:
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / num_batches))
            for layer in self.layers:
                logger.debug("%s", layer)

    def predict_set(self, inputs):
        nrecs = inputs.shape[inputs.major_axis()]
        outputs = self.backend.alloc(nrecs, self.layers[-1].nout)
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for layer in self.layers:
            layer.set_train_mode(False)
        for batch in xrange(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, nrecs)
            self.fprop(inputs.get_minor_slice(start_idx, end_idx))
            outputs.set_minor_slice(start_idx, end_idx, self.layers[-1].output)
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
                preds_shape = list(outputs.shape)
                preds_shape[outputs.minor_axis()] = 1
                preds['train'] = dataset.backend.empty(preds_shape)
                dataset.backend.argmax(outputs, axis=outputs.minor_axis(),
                                       out=preds['train'])
            if test and 'test' in inputs:
                outputs = self.predict_set(inputs['test'])
                preds_shape = list(outputs.shape)
                preds_shape[outputs.minor_axis()] = 1
                preds['test'] = dataset.backend.empty(preds_shape)
                dataset.backend.argmax(outputs, axis=outputs.minor_axis(),
                                       out=preds['test'])
            if validation and 'validation' in inputs:
                outputs = self.predict_set(inputs['validation'])
                preds_shape = list(outputs.shape)
                preds_shape[outputs.minor_axis()] = 1
                preds['validation'] = dataset.backend.empty(preds_shape)
                dataset.backend.argmax(outputs, axis=outputs.minor_axis(),
                                       out=preds['validation'])
            if len(preds) is 0:
                logger.error("must specify >=1 of: train, test, validation")
            res.append(preds)
        return res

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers:
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs, epoch):
        i = self.nlayers - 1
        error = self.cost.apply_derivative(self.backend, self.layers[i].output,
                                           targets, self.temp)
        batch_size = self.batch_size
        if self.dist_mode == 'datapar':
            batch_size *= MPI.COMM_WORLD.size
        self.backend.divide(error, self.backend.wrap(batch_size), out=error)

        while i > 0:
            self.layers[i].bprop(error, self.layers[i - 1].output, epoch)
            error = self.layers[i].berror
            i -= 1
        self.layers[i].bprop(error, inputs, epoch)

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
                    misclass = ds.backend.empty(preds[item].shape)
                    ds.backend.argmax(targets[item],
                                      axis=targets[item].minor_axis(),
                                      out=misclass)
                    ds.backend.not_equal(preds[item], misclass, misclass)
                    self.result = ds.backend.mean(misclass)
                    logging.info("%s set misclass rate: %0.5f%%" % (
                        item, 100 * self.result))
        # TODO: return values instead?
