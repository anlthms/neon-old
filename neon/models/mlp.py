# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple multi-layer perceptron model.
"""

import logging
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
        self.result = 0
        self.data_layer = self.layers[0]
        self.cost_layer = self.layers[-1]
        self.class_layer = self.layers[-2]

    def link(self, initlayer=None):
        for ll, pl in zip(self.layers, [initlayer] + self.layers[:-1]):
            ll.set_previous_layer(pl)

    def initialize(self, backend, initlayer=None):
        if self.initialized:
            return
        self.backend = backend
        kwargs = {"backend": self.backend, "batch_size": self.batch_size,
                  "accumulate": self.accumulate}
        for ll, pl in zip(self.layers, [initlayer] + self.layers[:-1]):
            ll.initialize(kwargs)
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
        for layer in self.layers:
            printfunc("%s", str(layer))

    def update(self, epoch):
        for layer in self.layers:
            layer.update(epoch)

    def get_classifier_output(self):
        return self.class_layer.output

    def print_training_error(self, error, num_batches, partial=False):
        rederr = self.backend.reduce_cost(error)
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

    def print_test_error(self, setname, misclass, logloss, nrecs):
        redmisclass = self.backend.reduce_cost(misclass)
        redlogloss = self.backend.reduce_cost(logloss)
        if self.backend.rank() != 0:
            return

        misclassval = redmisclass / nrecs
        loglossval = redlogloss / nrecs
        self.result = misclassval
        logging.info("%s set misclass rate: %0.5f%% logloss %0.5f",
                     setname, 100 * misclassval, loglossval)

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
            self.print_test_error(setname, misclass_sum, logloss_sum, nrecs)
            self.data_layer.cleanup()
            return_err[setname] = self.result
        return return_err


class MLPL(MLP):
    """
    Localization model. Inherits everythning from MLPB that does the learning
    of the features. Then runs a forward pass on the larger images and [todo]
    plots localization maps.
    """

    def predict_and_localize(self, dataset=None):

        # setting up data
        if dataset is not None:
            self.data_layer.init_dataset(dataset)
        dataset.set_batch_size(self.batch_size)
        self.data_layer.use_set('validation', predict=True)

        # seting up layers
        self.layers[0].ofmshape = [32, 32]  # TODO: Move this to yaml

        for l in range(1, len(self.layers)-1):
            delattr(self.layers[l], 'delta_shape')
            delattr(self.layers[l], 'out_shape')

        self.link()
        self.initialize(self.backend)
        self.print_layers()
        self.fprop()
        self.visualize()

    def visualize(self):
        """
        Rudimentary visualization code for localization experiments:
        """
        from ipdb import set_trace as trace
        import matplotlib.pyplot as plt
        # look at the data
        mapp = self.layers[6].output.asnumpyarray()  # 50 is 2 x (5x5)
        mapp0 = mapp[0:25].reshape(5, 5, -1)
        mapp1 = mapp[25:50].reshape(5, 5, -1)
        databatch = self.layers[0].output.asnumpyarray()
        data0 = databatch[0*1024:1*1024].reshape(32, 32, -1)
        data1 = databatch[1*1024:2*1024].reshape(32, 32, -1)

        self.myplot(plt, mapp0, title='positive class label strength',
                    span=(0, 1), fig=0)
        self.myplot(plt, mapp1, title='negative class label strength',
                    span=(0, 1), fig=1)
        self.myplot(plt, data0, title='data variable 0',
                    span=(-1, 1.5), fig=2)
        self.myplot(plt, data1, title='data variable 1', span=(-2, 2), fig=3)

        print("setting trace to keep plots open...")
        trace()

    @staticmethod
    def myplot(plt, data, title, span, fig):
        """
        wrapper for imshow that goes through 100 examples and makes subplots.
        TODO: Move this and visualize() to diagnostics.
        """
        plt.figure(fig)
        plt.clf()
        for i in range(100):
            plt.subplot(10, 10, i+1)
            plt.imshow(data[..., i], interpolation='none',
                       vmin=span[0], vmax=span[1])
        plt.subplot(10, 10, 5)
        plt.title(title)
