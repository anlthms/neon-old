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
from ipdb import set_trace as trace

logger = logging.getLogger(__name__)


class RNN(Model):

    """
    Recurrent neural network. Supports LSTM and standard RNN layers.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for req_param in ['layers', 'batch_size']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)
        self.nlayers = len(self.layers)
        self.cost.initialize(kwargs)

    def fit(self, dataset):
        self.dataset = dataset
        # self.grad_checker(numgrad="lstm_fx")
        # pick one: for standard rnn, "output" "input" "rec"
        # for LSTM, "output" "lstm_ix" "lstm_ih" "lstm_fh" "...

        """
        Learn model weights on the given dataset.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))
        inputs = dataset.get_inputs(train=True)['train']
        # use targets = inputs for sequence prediction
        targets = self.backend.copy(inputs)
        nrecs = inputs.shape[0]
        nin = self.layers[0].nin
        if self.make_plots:
            viz = VisualizeRNN()
        num_batches = nrecs / nin / self.unrolls - 1
        logger.info('Divide input %d into batches of size %d with %d timesteps'
                    'for %d batches',
                    nrecs, self.batch_size, self.unrolls, num_batches)
        logger.info('commencing model fitting')
        suberrorlist = []
        errorlist = []
        error = self.backend.empty((1, 1))
        suberror = self.backend.empty((1, 1))
        hidden_init = self.backend.zeros((self.layers[1].nin,
                                          self.batch_size))
        cell_init = self.backend.zeros((self.layers[1].nin,
                                        self.batch_size))
        for epoch in range(self.num_epochs):
            error.fill(0)
            suberror.fill(0)
            hidden_init.fill(0)
            cell_init.fill(0)
            for batch in xrange(num_batches):
                startidx = batch*nin*self.unrolls
                endidx = (batch+1)*nin*(self.unrolls+1)
                cur_input = inputs[startidx:endidx]
                cur_tgt = targets[startidx:endidx]
                cur_tgt_out = cur_tgt[self.unrolls*nin:(self.unrolls+1)*nin]
                self.fprop(cur_input, hidden_init, cell_init)
                self.bprop(cur_tgt, cur_input)
                self.update(epoch)

                hidden_init = self.layers[0].output_list[-1]
                if 'c_t' in self.layers[0].__dict__:
                    cell_init = self.layers[0].c_t[-1]
                if (batch % self.reset_period) == 0:  # reset hidden state
                    hidden_init.fill(0)
                    if 'c_t' in self.layers[0].__dict__:
                        cell_init.fill(0)
                self.cost.set_outputbuf(self.layers[-1].output_list[-1])

                suberror = self.cost.apply_function(cur_tgt_out)
                self.backend.divide(suberror, float(self.batch_size*nin),
                                    suberror)
                suberrorlist.append(float(suberror.asnumpyarray()))
                self.backend.divide(suberror, num_batches, suberror)
                self.backend.add(error, suberror, error)
            errorlist.append(float(error.asnumpyarray()))
            if self.make_plots is True:
                viz.plot_weights(self.layers[0].weights.asnumpyarray(),
                                 self.layers[0].Wih.asnumpyarray(),
                                 self.layers[1].weights.asnumpyarray())
                viz.plot_lstm_wts(self.layers[0], scale=2.1, fig=4)
                viz.plot_lstm_acts(self.layers[0], scale=2.1, fig=5)

                viz.plot_error(suberrorlist, errorlist)
                viz.plot_activations(self.layers[0].net_i,
                                     self.layers[0].i_t,
                                     self.layers[1].pre_act_list,
                                     self.layers[1].output_list,
                                     cur_tgt)
            logger.info('epoch: %d, total training error per element: %0.5f',
                        epoch, error.asnumpyarray())
            for layer in self.layers:
                logger.debug("%s", layer)

    def grad_checker(self, numgrad="lstm_ch"):
        """
        Check gradients for LSTM layer:
          - W is replicated, only inject the eps once, repeat, average.
            bProp is only through the full stack, but wrt. the W in each
            level. bProp does this through a for t in tau.

            Need a special fprop that injects into one unrolling only.
        """
        for layer in self.layers:
            logger.info("%s", str(layer))
        inputs = self.dataset.get_inputs(train=True)['train']
        nin = self.layers[0].nin
        # use targets = inputs for sequence prediction
        targets = self.backend.copy(inputs)
        nrecs = inputs.shape[0]  # was shape[1], moved to new dataset format
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        batch = 0

        startidx = batch*nin*self.unrolls
        endidx = (batch+1)*nin*(self.unrolls+1)
        cur_input = inputs[startidx:endidx]
        cur_tgt = targets[startidx:endidx]
        cur_tgt_out = cur_tgt[self.unrolls*nin:(self.unrolls+1)*nin]

        hidden_init = self.backend.zeros((self.layers[1].nin, self.batch_size))
        cell_init = self.backend.zeros((self.layers[1].nin, self.batch_size))

        if numgrad is "output":
            num_target = self.layers[1].weights
            an_target = self.layers[1].weight_updates
            num_i, num_j = 15, 56
        elif numgrad is "input":
            num_target = self.layers[0].weights
            an_target = self.layers[0].weight_updates
            num_i, num_j = 12, 110  # 110 is "n"
        elif numgrad is "rec":
            num_target = self.layers[0].weights_rec
            an_target = self.layers[0].updates_rec
            num_i, num_j = 12, 63
        elif numgrad.startswith("lstm"):
            gate = numgrad[-2:]
            num_target = getattr(self.layers[0], 'W'+gate)
            an_target = getattr(self.layers[0], 'W'+gate+'_updates')
            if "x" in numgrad:
                num_i, num_j = 12, 110
            if "h" in numgrad:
                num_i, num_j = 12, 55

        eps = 1e-6  # better to use float64 in cpu.py for this
        numerical = 0  # initialize buffer
        # extra loop to inject epsilon in different unrolling stages
        for tau in range(0, self.unrolls):
            # reset state
            hidden_init.fill(0)
            cell_init.fill(0)
            # fprop with eps
            self.fprop_eps(cur_input, tau, eps, hidden_init, cell_init,
                           num_target=num_target, num_i=num_i, num_j=num_j)
            self.cost.set_outputbuf(self.layers[-1].output_list[-1])
            suberror_eps = self.cost.apply_function(cur_tgt_out).asnumpyarray()

            self.fprop_eps(cur_input, tau, 0, hidden_init, cell_init,
                           num_target=num_target, num_i=num_i, num_j=num_j)
            self.cost.set_outputbuf(self.layers[-1].output_list[-1])
            suberror_ref = self.cost.apply_function(cur_tgt_out).asnumpyarray()
            num_part = (suberror_eps - suberror_ref) / eps / \
                float(self.batch_size * nin)
            logger.info("numpart for  tau=%d of %d is %e",
                        tau, self.unrolls, num_part)
            numerical += num_part

        # bprop for comparison
        self.bprop(cur_tgt, cur_input, numgrad=numgrad)

        analytical = an_target[num_i, num_j].asnumpyarray()
        logger.info("RNN grad_checker: suberror_eps %f", suberror_eps)
        logger.info("RNN grad_checker: suberror_ref %f", suberror_ref)
        logger.info("RNN grad_checker: numerical %s %e", numgrad, numerical)
        logger.info("RNN grad_checker: analytical %s %e", numgrad, analytical)
        logger.info("RNN grad_checker: ratio %e", numerical/analytical)

    def fprop_eps(self, inputs, eps_tau, eps, hidden_init,
                  cell_init, unrolls=None,
                  num_target=None, num_i=0, num_j=0):
        """
        have a pre_act and output for every unrolling step. The layer needs
        to keep track of all of these, so we tell it which unroll we are in.
        """
        nin = self.layers[0].nin

        if unrolls is None:
            unrolls = self.unrolls
        y = hidden_init
        c = cell_init

        for tau in range(0, unrolls):
            if tau == eps_tau:
                num_target[num_i, num_j] = (num_target[num_i,
                                                       num_j].asnumpyarray() +
                                            eps)

            self.layers[0].fprop(y=y, inputs=inputs[nin*tau:nin*(tau+1), :],
                                 tau=tau, cell=c)
            y = self.layers[0].output_list[tau]
            if 'c_t' in self.layers[0].__dict__:
                c = self.layers[0].c_t[tau]
            self.layers[1].fprop(inputs=y, tau=tau)

            if tau == eps_tau:
                num_target[num_i, num_j] = (num_target[num_i,
                                                       num_j].asnumpyarray() -
                                            eps)

    def fprop(self, inputs, hidden_init,
              cell_init, unrolls=None):
        """
        have a pre_act and output for every unrolling step. The layer needs
        to keep track of all of these, so we tell it which unroll we are in.
        """
        nin = self.layers[0].nin

        if unrolls is None:
            unrolls = self.unrolls
        y = hidden_init
        c = cell_init
        # fprop does a single full unroll
        for tau in range(0, unrolls):
            self.layers[0].fprop(y=y, inputs=inputs[nin*tau:nin*(tau+1), :],
                                 tau=tau, cell=c)
            y = self.layers[0].output_list[tau]
            if 'c_t' in self.layers[0].__dict__:
                c = self.layers[0].c_t[tau]
            self.layers[1].fprop(inputs=y, tau=tau)

    def bprop(self, targets, inputs, numgrad=None):
        """
        Refactor:
        This bprop has an OUTER FOR LOOP over t-BPTT unrollings
            for a given unrolling depth, we go output-hidden-hidden-input
            which breaks down as:
                  layers[1].bprop -- output layer

        """
        nin = self.layers[0].nin

        if numgrad is None:
            min_unroll = 1
        else:
            min_unroll = self.unrolls

        # this loop is a property of t-BPTT through different depth.
        # inside this loop, go through the input-hidden-output stack.
        for tau in range(min_unroll, self.unrolls+1):

            # output layers[1]:
            self.cost.set_outputbuf(self.layers[1].output_list[tau - 1])
            error = self.cost.apply_derivative(targets[nin*tau:nin*(tau+1), :])
            esize = error.shape[0] * error.shape[1]
            self.backend.divide(error, esize, out=error)
            self.layers[1].bprop(error, self.layers[0].output_list[tau - 1],
                                 tau, numgrad)

            # recurrent layers[0]: loop over different unrolling sizes
            error_h = self.layers[1].berror
            error_c = self.backend.zeros((self.layers[1].nin,
                                          self.batch_size))
            for t in list(range(0, tau))[::-1]:  # restored to 0 as in old RNN
                self.layers[0].bprop(error_h, error_c, inputs, tau, t, numgrad)
                error_h = self.backend.copy(self.layers[0].berror)
                if 'cerror' in self.layers[0].__dict__:
                    error_c = self.layers[0].cerror

    def update(self, epoch):
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

        outputs are computed as a 2000 x self.batch_size matrix that is then
        flattened to return_buffer of 100000 records. Used for preds['train']
        """
        be = self.backend
        nrecs = inputs.shape[0]
        nin = self.layers[0].nin
        num_batches = int(math.floor((nrecs) / nin / self.unrolls)) - 1
        outputs = be.zeros((num_batches*(self.unrolls), self.batch_size))
        hidden_init = be.zeros((self.layers[1].nin, self.batch_size))
        cell_init = be.zeros((self.layers[1].nin, self.batch_size))
        letters = be.empty((1, self.batch_size), dtype='int32')

        for batch in xrange(num_batches):
            startidx = batch*nin*self.unrolls
            endidx = (batch+1)*nin*(self.unrolls+1)
            cur_input = inputs[startidx:endidx]
            self.fprop(cur_input, hidden_init, cell_init, unrolls=self.unrolls)
            hidden_init = self.layers[0].output_list[-1]
            if 'c_t' in self.layers[0].__dict__:
                    cell_init = self.layers[0].c_t[-1]
            if (batch % self.reset_period) == 0:
                    hidden_init.fill(0)
                    if 'c_t' in self.layers[0].__dict__:
                        cell_init.fill(0)
            for tau in range(self.unrolls):
                be.argmax(self.layers[1].output_list[tau], axis=0, out=letters)
                idx = (self.unrolls)*batch + tau
                outputs[idx, :] = letters

        return_buffer = be.zeros(((num_batches+1)*self.unrolls,
                                 self.batch_size))
        return_buffer[0:num_batches*self.unrolls, :] = outputs
        return return_buffer

    def predict(self, train=True, test=True, validation=False):
        """
        Iterate over data sets and call predict_set for each.
        This is called directly from the fit_predict_err experiment.

        Returns:
            res: a list of (key,value) pairs, e.g. res[0]['train'] is a tensor
                 of class labels
        """
        ds = self.dataset
        inputs = ds.get_inputs(train=train, test=test)
        preds = dict()
        if train and 'train' in inputs:
            preds['train'] = self.predict_set(inputs['train'])
        if test and 'test' in inputs:
            preds['test'] = self.predict_set(inputs['test'])
        if validation and 'validation' in inputs:
            preds['validation'] = self.predict_set(inputs['validation'])
        if len(preds) == 0:
            logger.error("must specify >=1 of: train, test, validation")
        return preds

    def error_metrics(self, ds, preds, train=True, test=True,
                      validation=False):
        """
        Iterate over predictions from predict() and compare to the targets.
        Targets come from dataset. [Why in a separate function?]
        """
        be = self.backend
        items = []
        if train:
            items.append('train')
        if test:
            items.append('test')
        if validation:
            items.append('validation')

        nin = self.layers[0].nin
        targets = ds.get_inputs(train=True, test=True, validation=False)
        targets['train'] = targets['train'][nin:]
        targets['test'] = targets['test'][nin:]
        self.result = be.empty((1, 1))
        for item in items:
            if item in targets and item in preds:
                num_batches = targets[item].shape[0] / nin / 5 * 5
                # misclass = be.zeros(num_batches * nin)
                misclass = be.zeros((num_batches, self.batch_size),
                                    dtype='int32')
                for i in range(num_batches):
                    be.argmax(targets[item][i * nin:(i + 1) * nin],
                              axis=0, out=misclass[i])
                import numpy as np
                tmp = misclass[:18, 30].asnumpyarray().astype(np.int8).T
                logging.info("the target for %s is %s", item, tmp.view('c'))
                tmp = preds[item][:18, 30].asnumpyarray().astype(np.int8).T
                logging.info("prediction for %s is %s", item, tmp.view('c'))
                be.not_equal(preds[item][:num_batches], misclass, misclass)
                be.mean(misclass, axes=None, out=self.result)
                logging.info("%s set misclass rate: %0.5f%%", item,
                             100 * self.result.asnumpyarray())
        # TODO: return values instead?
        if self.make_plots:
            trace()  # just used to keep figures open
