# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple multi-layer perceptron model.
"""

import logging
import numpy as np
from neon.models.model import Model
from neon.util.param import opt_param, req_param

logger = logging.getLogger(__name__)


class MLP(Model):

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

    def link_and_initialize(self, layer_list, kwargs, initlayer=None):
        for ll, pl in zip(layer_list, [initlayer] + layer_list[:-1]):
            self.backend.begin()
            ll.set_previous_layer(pl)
            ll.initialize(kwargs)
            self.backend.end()

    def fprop(self):
        for ll, pl in zip(self.layers, [None] + self.layers[:-1]):
            self.backend.begin()
            y = None if pl is None else pl.output
            ll.fprop(y)
            self.backend.end()

    def bprop(self):
        for ll, nl in zip(reversed(self.layers),
                          reversed(self.layers[1:] + [None])):
            self.backend.begin()
            error = None if nl is None else nl.deltas
            ll.bprop(error)
            self.backend.end()

    def print_layers(self, debug=False):
        printfunc = logger.debug if debug else logger.info
        for layer in self.layers:
            self.backend.begin()
            printfunc("%s", str(layer))
            self.backend.end()

    def update(self, epoch):
        for layer in self.layers:
            self.backend.begin()
            layer.update(epoch)
            self.backend.end()

    def get_classifier_output(self):
        return self.class_layer.output

    def fit(self, dataset):
        """
        Learn model weights on the given datasets.
        """
        error = self.backend.zeros((1, 1))
        self.print_layers()
        self.data_layer.init_dataset(dataset)
        self.data_layer.use_set('train')
        logger.info('commencing model fitting')
        while self.epochs_complete < self.num_epochs:
            self.backend.begin()
            error.fill(0.0)
            mb_id = 1
            self.data_layer.reset_counter()
            while self.data_layer.has_more_data():
                self.backend.begin()
                self.fprop()
                self.bprop()
                self.update(self.epochs_complete)
                self.backend.add(error, self.cost_layer.get_cost(), error)
                if self.step_print > 0 and mb_id % self.step_print == 0:
                    logger.info('%d.%d logloss=%0.5f', self.epochs_complete,
                                mb_id / self.step_print - 1,
                                np.int(error.asnumpyarray()) / mb_id)
                mb_id += 1
                self.backend.end()
            logger.info('epoch: %d, total training error: %0.5f',
                        self.epochs_complete,
                        error.asnumpyarray() / self.data_layer.num_batches)
            self.print_layers(debug=True)
            self.epochs_complete += 1
            self.backend.end()
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

        return_err = dict()

        for setname in ['train', 'test', 'validation']:
            self.backend.begin()
            if self.data_layer.has_set(setname) is False:
                self.backend.end()
                continue
            self.data_layer.use_set(setname, predict=True)
            self.data_layer.reset_counter()
            misclass_sum.fill(0.0)
            logloss_sum.fill(0.0)
            nrecs = self.batch_size * self.data_layer.num_batches
            while self.data_layer.has_more_data():
                self.backend.begin()
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
                self.backend.end()
            logging.info("%s set misclass rate: %0.5f%% logloss %0.5f" % (
                setname, 100 * misclass_sum.asnumpyarray() / nrecs,
                logloss_sum.asnumpyarray() / nrecs))
            self.result = misclass_sum.asnumpyarray()[0, 0] / nrecs
            self.data_layer.cleanup()
            return_err[setname] = self.result
            self.backend.end()
        return return_err
