# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple recurrent neural network with one hidden layer.
"""

import logging
import math

from neon.diagnostics.visualize_rnn import VisualizeRNN
from neon.models.model import Model
from neon.util.compat import range

logger = logging.getLogger(__name__)


class RNN(Model):

    """
    Recurrent neural network
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for req_param in ['layers']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)
        self.nlayers = len(self.layers)

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = inputs.copy()  # use targets = inputs for sequence prediction
        nrecs = inputs.shape[0]  # was shape[1], moved to new dataset format
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        viz = VisualizeRNN()
        num_batches = int(math.floor((nrecs + 0.0) / self.batch_size
                                                   / self.unrolls)) - 1
        logger.info('Divide input %d into batches of size %d with %d timesteps'
                    'for %d batches',
                    nrecs, self.batch_size, self.unrolls, num_batches)
        logger.info('commencing model fitting')
        suberrorlist = []
        errorlist = []
        for epoch in range(self.num_epochs):
            error = 0
            suberror = self.backend.zeros(num_batches)
            batch_inx = self.backend.zeros((self.batch_size,
                                            self.unrolls+1), dtype=int)
            hidden_init = self.backend.zeros((self.batch_size,
                                             self.layers[1].nin))
            for batch in range(num_batches):
                self.serve_batch(batch, batch_inx, num_batches)  # get indices
                self.fprop(inputs, batch_inx, hidden_init,
                           debug=(True if batch == -1 else False))
                self.bprop(targets, inputs, batch_inx, epoch)
                hidden_init = self.layers[0].output_list[-1]
                if batch % 20 is 0:  # reset hidden state periodically
                    hidden_init = self.backend.zeros((self.batch_size,
                                                     self.layers[1].nin))
                self.cost.set_outputbuf(self.layers[-1].output_list[-1])
                suberror = self.cost.apply_function(targets[batch_inx[:, -1]])
                suberror /= float(self.batch_size * self.layers[0].nin)
                suberrorlist.append(suberror)
                error += suberror / num_batches
            errorlist.append(error)
            if self.make_plots is True:
                viz.plot_weights(self.layers[0].weights.raw(),
                                 self.layers[0].weights_rec.raw(),
                                 self.layers[1].weights.raw())
                viz.plot_error(suberrorlist, errorlist)
                viz.plot_activations(self.layers[0].pre_act_list,
                                     self.layers[0].output_list,
                                     self.layers[1].pre_act_list,
                                     self.layers[1].output_list,
                                     targets, batch_inx)
            logger.info('epoch: %d, total training error per element: %0.5f' %
                        (epoch, error))
            for layer in self.layers:
                logger.debug("%s", layer)

        # print the prediction
        self.visualize(datasets, viz)

    def serve_batch(self, batch, batch_inx, num_batches):
        """
        For t-BPTT, need the batches to be layed out like this:
        (For 1000 batches, 3 unrolls, batch size 50)
        2,999 5,999  ...  149,999
          .
          .
        0,003
        ---------------------      ^  1000 batches
        0,002 3,002
        0,001 3,001                ^  3  unrolls
        0,000 3,000  ...  147,000  -> 50 batch size

        Each SGD step perform BPTT through a 50x3 block of this data. Because
        this corresponds to a 128x50x3 tensor, we only pass indices and not
        the minibatch directly.
        This function returns the submatrix delimited by the --- and constructs
        it row by row. Wtih batch size 50 and 3 unrolls, constructing 1000
        batches takes 150k data points.

        Inputs:
            batch: batch number
            batch_inx: buffer passed for index emptyation

        Returns:
            updates the buffer batch_inx directly to minimize emptyations.
        """

        for tau in range(self.unrolls+1):
            batch_inx[:, tau] = (self.unrolls*batch + tau
                                 + self.unrolls*num_batches
                                 * self.backend.array(range(self.batch_size)))

    def visualize(self, datasets, viz):
        """
        Generate and print predictions on the given datasets.
        """
        inputs = datasets[0].get_inputs(test=True)['test']
        # targets = datasets[0].get_targets(test=True)['test']
        batch_inx = self.backend.zeros((self.batch_size, self.unrolls+1),
                                       dtype=int)
        for i in range(self.unrolls+1):
            batch_inx[:, i] = self.backend.array(range(i, i+self.batch_size),
                                                 dtype=int)
        hidden_init = self.backend.zeros((self.batch_size, self.layers[1].nin))
        self.fprop(inputs, batch_inx, hidden_init)

        viz.print_text(inputs[self.unrolls:self.unrolls+50, :],
                       self.layers[1].output_list[self.unrolls-1])

    def fprop(self, inputs, batch_inx, hidden_init, debug=False, unrolls=None):
        """
        have a pre_act and output for every unrolling step. The layer needs
        to keep track of all of these, so we tell it which unroll we are in.
        """
        if unrolls is None:
            unrolls = self.unrolls
        if debug:
            import numpy as np
            print "fprop input", np.nonzero(inputs[0:self.unrolls, :].raw())[1]
        y = hidden_init
        for tau in range(0, unrolls):
            self.layers[0].fprop(y, inputs[batch_inx[:, tau], :], tau)
            y = self.layers[0].output_list[tau]
            self.layers[1].fprop(y, tau)

    def bprop(self, targets, inputs, batch_inx, epoch, debug=False):

        full_unroll = True
        if full_unroll:
            min_unroll = 1
        else:
            min_unroll = self.unrolls

        # clear deltas
        for tau in range(min_unroll, self.unrolls+1):
            self.backend.fill(self.layers[0].deltas[tau], 0)
            self.backend.fill(self.layers[1].deltas_o[tau], 0)
        # FOUND BUG: also should clear updates
        self.backend.fill(self.layers[0].weight_updates, 0)
        self.backend.fill(self.layers[0].updates_rec, 0)
        self.backend.fill(self.layers[1].weight_updates, 0)

        # fill deltas
        for tau in range(min_unroll, self.unrolls+1):
            if debug:
                import numpy as np
                print "unrolling", tau, "of", self.unrolls
                print "in bprop, input", np.nonzero(inputs[0:tau, :].raw())[1]
                print "backprop target", np.flatnonzero(targets[tau, :].raw())
            self.cost.set_outputbuf(self.layers[1].output_list[tau - 1])
            error = self.cost.apply_derivative(targets[batch_inx[:, tau]])
            error /= float(error.shape[0] * error.shape[1])
            self.layers[1].bprop(error,
                                 self.layers[0].output_list[tau - 1],
                                 tau)

        cerror = self.backend.zeros((self.batch_size, self.layers[0].nout))
        for tau in range(min_unroll, self.unrolls+1):
            # need to bprop from the output layer before calling bprop
            self.backend.bprop_fc(self.layers[1].weights,
                                  self.layers[1].deltas_o[tau].transpose(),
                                  out=cerror)
            self.layers[0].bprop(cerror, inputs, tau, batch_inx)

        # apply updates
        for layer in self.layers:
            layer.update(epoch)

    def predict_set(self, inputs):
        """
        compute predictions for a set of inputs. This does the actual work.
        The tricky bit is that with how batches are sliced, we have a block of
        predictions that needs to be shaped back into a long vector. This
        is done by having a for loop over batches load a matrix, that is
        flattened at the end (each batch has a non-contigous access pattern
        with respect to the full dataset.)
        """
        nrecs = inputs.shape[0]
        num_batches = int(math.floor((nrecs) / self.batch_size
                                             / self.unrolls)) - 1
        outputs = self.backend.zeros((num_batches*(self.unrolls),
                                      self.batch_size))
        hidden_init = self.backend.zeros((self.batch_size, self.layers[1].nin))
        batch_inx = self.backend.zeros((self.batch_size,
                                        self.unrolls+1), dtype=int)
        for batch in range(num_batches):
            self.serve_batch(batch, batch_inx, num_batches)
            self.fprop(inputs, batch_inx, hidden_init, unrolls=self.unrolls)
            hidden_init = self.layers[0].output_list[-1]
            if batch % 20 is 0:
                    hidden_init = self.backend.zeros((self.batch_size,
                                                     self.layers[1].nin))
            for tau in range(self.unrolls):
                letters = self.backend.empty(50, dtype=int)
                self.backend.argmax(self.layers[1].output_list[tau],
                                    axis=1, out=letters)
                idx = (self.unrolls)*batch + tau
                outputs[idx, :] = letters
        return_buffer = self.backend.zeros(nrecs)
        nrecs_eff = num_batches*self.unrolls*self.batch_size
        return_buffer[0:nrecs_eff] = outputs.transpose().reshape((-1,))
        return return_buffer

    def predict(self, datasets, train=True, test=True, validation=False):
        """
        Iterate over data sets and call predict_set for each.
        This is called directly from the fit_predict_err experiment.

        Returns:
            res: a list of (key,value) pairs, e.g. res[0]['train'] is a tensor
                 of class labels
        """
        res = []
        for dataset in datasets:
            inputs = dataset.get_inputs(train=train, test=test)
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

    def error_metrics(self, datasets, predictions, train=True, test=True,
                      validation=False):
        """
        Iterate over predictions from predict() and compare to the targets.
        Targets come from dataset. [Why in a separate function?]
        """
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
            targets = ds.get_targets(train=True, test=True, validation=False)
            for item in items:
                print "item:", item
                if item in targets and item in preds:
                    misclass = ds.backend.empty(preds[item].shape)
                    ds.backend.argmax(targets[item],
                                      axis=1,
                                      out=misclass)
                    ds.backend.not_equal(preds[item], misclass, misclass)
                    self.result = ds.backend.mean(misclass)
                    logging.info("%s set misclass rate: %0.5f%%" % (
                        item, 100 * self.result))
        # TODO: return values instead?
