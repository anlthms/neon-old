# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple multi-layer perceptron model.
"""

import logging
import neon.util.metrics as ms
from neon.models.deprecated.mlp import MLP as MLP_old  # noqa
from neon.util.param import opt_param, req_param

logger = logging.getLogger(__name__)


class MLP(MLP_old):

    """
    Fully connected, feed-forward, multi-layer perceptron model
    """

    def __init__(self, **kwargs):
        self.initialized = False
        self.dist_mode = None
        self.__dict__.update(kwargs)
        req_param(self, ['layers', 'batch_size'])
        opt_param(self, ['step_print'], -1)
        opt_param(self, ['accumulate'], False)
        opt_param(self, ['reuse_deltas'], True)
        self.result = 0
        self.data_layer = self.layers[0]
        self.cost_layer = self.layers[-1]
        self.class_layer = self.layers[-2]

    def link(self, initlayer=None):
        for ll, pl in zip(self.layers, [initlayer] + self.layers[:-1]):
            ll.set_previous_layer(pl)
        self.print_layers()

    def initialize(self, backend, initlayer=None):
        if self.initialized:
            return
        self.backend = backend
        kwargs = {"backend": self.backend, "batch_size": self.batch_size,
                  "accumulate": self.accumulate}
        for ll, pl in zip(self.layers, [initlayer] + self.layers[:-1]):
            ll.initialize(kwargs)

        self.nin_max = max(map(lambda x: x.nin, self.layers[1:-1]))
        self.global_deltas = None
        if self.reuse_deltas:
            self.global_deltas = backend.zeros(
                (2 * self.nin_max, self.batch_size),
                self.layers[1].deltas_dtype)

        for idx, ll in enumerate(self.layers[1:-1]):
            ll.set_deltas_buf(self.global_deltas,
                              offset=((idx % 2) * self.nin_max))

        self.initialized = True

    def fprop(self):
        for ll, pl in zip(self.layers, [None] + self.layers[:-1]):
            y = None if pl is None else pl.output
            ll.fprop(y)

    def bprop(self):
        for ll, nl in zip(reversed(self.layers),
                          reversed(self.layers[1:] + [None])):
            error = None if nl is None else nl.deltas
            ll.bprop(error)

    def print_layers(self, debug=False):
        printfunc = logger.debug if debug else logger.info
        netdesc = 'Layers:\n'
        for layer in self.layers:
            netdesc += '\t' + str(layer) + '\n'
        printfunc("%s", netdesc)

    def update(self, epoch):
        for layer in self.layers:
            layer.update(epoch)

    def get_classifier_output(self):
        return self.class_layer.output

    def print_training_error(self, error, num_batches, partial=False):
        rederr = self.backend.reduce_tensor(error)
        if self.backend.rank() != 0:
            return

        errorval = rederr / num_batches
        if partial is True:
            assert self.step_print != 0
            logger.info('%d.%d training error: %0.5f', self.epochs_complete,
                        num_batches / self.step_print - 1, errorval)
        else:
            logger.info('epoch: %d, training error: %0.5f',
                        self.epochs_complete, errorval)

    def print_test_error(self, setname, misclass, nrecs):
        redmisclass = self.backend.reduce_tensor(misclass)
        if self.backend.rank() != 0:
            return

        misclassval = redmisclass / nrecs
        self.result = misclassval
        logging.info("%s set misclass rate: %0.5f%%",
                     setname, 100 * misclassval)

    def fit(self, dataset):
        """
        Learn model weights on the given datasets.
        """
        error = self.backend.zeros((1, 1))
        self.data_layer.init_dataset(dataset)
        self.data_layer.use_set('train')
        logger.info('commencing model fitting')
        while self.epochs_complete < self.num_epochs:
            error.fill(0.0)
            mb_id = 1
            self.data_layer.reset_counter()
            while self.data_layer.has_more_data():
                self.fprop()
                self.bprop()
                self.update(self.epochs_complete)
                self.backend.add(error, self.cost_layer.get_cost(), error)
                if self.step_print > 0 and mb_id % self.step_print == 0:
                    self.print_training_error(error, mb_id, partial=True)
                mb_id += 1
            self.print_training_error(error, self.data_layer.num_batches)
            self.print_layers(debug=True)
            self.epochs_complete += 1
        self.data_layer.cleanup()

    def predict_and_report(self, dataset=None):
        if dataset is not None:
            self.data_layer.init_dataset(dataset)
        predlabels = self.backend.empty((1, self.batch_size))
        labels = self.backend.empty((1, self.batch_size))
        misclass = self.backend.empty((1, self.batch_size))
        misclass_sum = self.backend.empty((1, 1))
        batch_sum = self.backend.empty((1, 1))

        return_err = dict()

        for ll in self.layers:
            ll.set_train_mode(False)

        for setname in ['train', 'test', 'validation']:
            if self.data_layer.has_set(setname) is False:
                continue
            self.data_layer.use_set(setname, predict=True)
            self.data_layer.reset_counter()
            misclass_sum.fill(0.0)
            nrecs = self.batch_size * self.data_layer.num_batches
            while self.data_layer.has_more_data():
                self.fprop()
                probs = self.get_classifier_output()
                reference = self.cost_layer.get_reference()
                ms.misclass_sum(self.backend, reference, probs, predlabels,
                                labels, misclass, batch_sum)
                self.backend.add(misclass_sum, batch_sum, misclass_sum)
            self.print_test_error(setname, misclass_sum, nrecs)
            self.data_layer.cleanup()
            return_err[setname] = self.result
        return return_err

    def predict_fullset(self, dataset, setname):
        self.data_layer.init_dataset(dataset)
        assert self.data_layer.has_set(setname)
        self.data_layer.use_set(setname, predict=True)
        self.data_layer.reset_counter()
        nrecs = self.batch_size * self.data_layer.num_batches
        outputs = self.backend.empty((self.class_layer.nout, nrecs))
        if self.data_layer.has_labels:
            reference = self.backend.empty((1, nrecs))
        else:
            reference = self.backend.empty(outputs.shape)
        batch = 0

        for ll in self.layers:
            ll.set_train_mode(False)

        while self.data_layer.has_more_data():
            self.fprop()
            start = batch * self.batch_size
            end = start + self.batch_size
            outputs[:, start:end] = self.get_classifier_output()
            reference[:, start:end] = self.cost_layer.get_reference()
            batch += 1

        self.data_layer.cleanup()
        return outputs, reference

    def report(self, reference, outputs, metric):
        nrecs = outputs.shape[1]
        if metric == 'misclass rate':
            retval = self.backend.empty((1, 1))
            labels = self.backend.empty((1, nrecs))
            preds = self.backend.empty(labels.shape)
            misclass = self.backend.empty(labels.shape)
            ms.misclass_sum(self.backend, reference, outputs,
                            preds, labels, misclass, retval)
            misclassval = retval.asnumpyarray() / nrecs
            return misclassval * 100

        if metric == 'auc':
            return ms.auc(self.backend, reference[0], outputs[0])

        if metric == 'log loss':
            retval = self.backend.empty((1, 1))
            sums = self.backend.empty((1, outputs.shape[1]))
            temp = self.backend.empty(outputs.shape)
            ms.logloss(self.backend, reference, outputs, sums, temp, retval)
            self.backend.multiply(retval, -1, out=retval)
            self.backend.divide(retval, nrecs, out=retval)
            return retval.asnumpyarray()

        raise NotImplementedError('metric not implemented:', metric)
