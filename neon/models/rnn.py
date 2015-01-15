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
        # self.grad_checker(numgrad="rec")
        # pick one: "output":"input":"rec"
        #           "lstm_x":"lstm_ih":"lstm_fh":"lstm_oh":"lstm_ch"

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
        viz = VisualizeRNN()
        num_batches = int(math.floor((nrecs + 0.0) / nin
                                                   / self.unrolls)) - 1
        logger.info('Divide input %d into batches of size %d with %d timesteps'
                    'for %d batches',
                    nrecs, self.batch_size, self.unrolls, num_batches)
        logger.info('commencing model fitting')
        suberrorlist = []
        errorlist = []
        error = self.backend.empty((1, 1))
        suberror = self.backend.empty(num_batches)
        for epoch in range(self.num_epochs):
            error.fill(0)
            suberror.fill(0)
            hidden_init = None
            cell_init = None
            for batch in xrange(num_batches):
                batch_inx = xrange(batch*nin*self.unrolls,
                                   (batch+1)*nin*self.unrolls+nin)
                self.fprop(inputs[batch_inx, :],
                           hidden_init=hidden_init, cell_init=cell_init,
                           debug=(True if batch == -1 else False))
                self.bprop(targets[batch_inx, :], inputs[batch_inx, :],
                           debug=(True if batch == -1 else False))
                self.update(epoch)
                hidden_init = self.layers[0].output_list[-1]
                if 'c_t' in self.layers[0].__dict__:
                    cell_init = self.layers[0].c_t[-1]
                if (batch % self.reset_period) == 0:  # reset hidden state
                    hidden_init.fill(0)
                    if 'c_t' in self.layers[0].__dict__:
                        cell_init.fill(0)
                self.cost.set_outputbuf(self.layers[-1].output_list[-1])
                target_out = targets[batch_inx, :][(self.unrolls-0)*nin:
                                                   (self.unrolls+1)*nin, :]
                suberror = self.cost.apply_function(target_out)
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
                import numpy as np
                viz.plot_lstm(self.layers[0].Wix.asnumpyarray(),
                              self.layers[0].Wfx.asnumpyarray(),
                              self.layers[0].Wox.asnumpyarray(),
                              self.layers[0].Wcx.asnumpyarray(),
                              np.hstack((self.layers[0].Wih.asnumpyarray(),
                                        self.layers[0].b_i.asnumpyarray(),
                                        self.layers[0].b_i.asnumpyarray())),
                              np.hstack((self.layers[0].Wfh.asnumpyarray(),
                                        self.layers[0].b_f.asnumpyarray(),
                                        self.layers[0].b_f.asnumpyarray())),
                              np.hstack((self.layers[0].Woh.asnumpyarray(),
                                        self.layers[0].b_o.asnumpyarray(),
                                        self.layers[0].b_o.asnumpyarray())),
                              np.hstack((self.layers[0].Wch.asnumpyarray(),
                                        self.layers[0].b_c.asnumpyarray(),
                                        self.layers[0].b_c.asnumpyarray())),
                              scale=1.1, fig=4)
                viz.plot_lstm(self.layers[0].i_t[0].asnumpyarray(),
                              self.layers[0].f_t[0].asnumpyarray(),
                              self.layers[0].o_t[0].asnumpyarray(),
                              self.layers[0].g_t[1].asnumpyarray(),
                              self.layers[0].net_i[0].asnumpyarray(),
                              self.layers[0].c_t[0].asnumpyarray(),
                              self.layers[0].c_t[1].asnumpyarray(),
                              self.layers[0].c_phi[1].asnumpyarray(),
                              scale=21, fig=5)
                viz.plot_error(suberrorlist, errorlist)
                viz.plot_activations(self.layers[0].net_i,
                                     self.layers[0].i_t,
                                     self.layers[1].pre_act_list,
                                     self.layers[1].output_list,
                                     targets[batch_inx, :])
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
        batch_inx = xrange(batch*self.layers[0].nin*self.unrolls,
                           (batch+1)*nin*self.unrolls+nin)
        target_out = targets[batch_inx, :][(self.unrolls-0)*nin:
                                           (self.unrolls+1)*nin, :]

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
        elif numgrad is "lstm_x":
            num_target = self.layers[0].Wfx
            an_target = self.layers[0].Wfx_updates
            num_i, num_j = 12, 110
        elif numgrad is "lstm_ih":
            num_target = self.layers[0].Wih
            an_target = self.layers[0].Wih_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_fh":
            num_target = self.layers[0].Wfh
            an_target = self.layers[0].Wfh_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_oh":
            num_target = self.layers[0].Woh
            an_target = self.layers[0].Woh_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_ch":
            num_target = self.layers[0].Wch
            an_target = self.layers[0].Wch_updates
            num_i, num_j = 12, 55

        eps = 1e-2  # better to use float64 in cpu.py for this
        numerical = 0  # initialize buffer
        # extra loop to inject epsilon in different unrolling stages
        for tau in range(0, self.unrolls):
            self.fprop_eps(inputs[batch_inx, :], tau, eps, hidden_init=None,
                           debug=(True if batch == -1 else False),
                           num_target=num_target, num_i=num_i, num_j=num_j)
            self.cost.set_outputbuf(self.layers[-1].output_list[-1])
            suberror_eps = self.cost.apply_function(target_out).asnumpyarray()

            self.fprop_eps(inputs[batch_inx, :], tau, 0, hidden_init=None,
                           debug=(True if batch == -1 else False),
                           num_target=num_target, num_i=num_i, num_j=num_j)
            self.cost.set_outputbuf(self.layers[-1].output_list[-1])
            suberror_ref = self.cost.apply_function(target_out).asnumpyarray()
            num_part = (suberror_eps - suberror_ref) / eps / \
                float(self.batch_size * nin)
            logger.info("numpart for  tau=%d of %d is %e",
                        tau, self.unrolls, num_part)
            numerical += num_part

        # bprop for comparison
        self.bprop(targets[batch_inx, :],
                   inputs[batch_inx, :], numgrad=numgrad)

        analytical = an_target[num_i, num_j].asnumpyarray()
        logger.info("RNN grad_checker: suberror_eps %f", suberror_eps)
        logger.info("RNN grad_checker: suberror_ref %f", suberror_ref)
        logger.info("RNN grad_checker: numerical %e", numerical)
        logger.info("RNN grad_checker: analytical %e", analytical)
        logger.info("RNN grad_checker: ratio %e", numerical/analytical)

    def fprop_eps(self, inputs, eps_tau, eps, hidden_init=None,
                  cell_init=None, debug=False, unrolls=None,
                  num_target=None, num_i=0, num_j=0):
        """
        have a pre_act and output for every unrolling step. The layer needs
        to keep track of all of these, so we tell it which unroll we are in.
        """
        nin = self.layers[0].nin

        if hidden_init is None:
            hidden_init = self.backend.zeros((self.layers[1].nin,
                                              self.batch_size))
        if cell_init is None:
            cell_init = self.backend.zeros((self.layers[1].nin,
                                            self.batch_size))
        if unrolls is None:
            unrolls = self.unrolls
        if debug:
            logger.info("fprop input\n%s",
                        str(inputs.reshape((6, nin, 50)).argmax(1)[:, 0:10]))
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

    def fprop(self, inputs, hidden_init=None,
              cell_init=None, debug=False, unrolls=None):
        """
        have a pre_act and output for every unrolling step. The layer needs
        to keep track of all of these, so we tell it which unroll we are in.
        """
        nin = self.layers[0].nin

        if hidden_init is None:
            hidden_init = self.backend.zeros((self.layers[1].nin,
                                              self.batch_size))
        if cell_init is None:
            cell_init = self.backend.zeros((self.layers[1].nin,
                                            self.batch_size))
        if unrolls is None:
            unrolls = self.unrolls
        if debug:
            logger.info("fprop input\n%s",
                        str(inputs.reshape((6, nin, 50)).argmax(1)[:, 0:10]))
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

    def bprop(self, targets, inputs, hidden_init=None, cell_init=None,
              debug=False, numgrad=None):
        """
        Refactor:
        This bprop has an OUTER FOR LOOP over t-BPTT unrollings
            for a given unrolling depth, we go output-hidden-hidden-input
            which breaks down as:
                  layers[1].bprop -- output layer

        """
        nin = self.layers[0].nin

        if hidden_init is None:
            hidden_init = self.backend.zeros((self.layers[1].nin,
                                              self.batch_size))
        if cell_init is None:
            cell_init = self.backend.zeros((self.layers[1].nin,
                                            self.batch_size))
        if numgrad is None:
            min_unroll = 1
        else:
            min_unroll = self.unrolls

        # [TODO] Move these to layer.update
        if 'weight_updates' in self.layers[0].__dict__:
            self.layers[0].weight_updates.fill(0)
        if 'updates_rec' in self.layers[0].__dict__:
            self.layers[0].updates_rec.fill(0)
        self.layers[1].weight_updates.fill(0)
        if 'Wix_updates' in self.layers[0].__dict__:
            # reset these things back to zero
            self.layers[0].Wix_updates.fill(0)
            self.layers[0].Wfx_updates.fill(0)
            self.layers[0].Wox_updates.fill(0)
            self.layers[0].Wcx_updates.fill(0)
            self.layers[0].Wih_updates.fill(0)
            self.layers[0].Wfh_updates.fill(0)
            self.layers[0].Woh_updates.fill(0)
            self.layers[0].Wch_updates.fill(0)
            self.layers[0].b_i_updates.fill(0)
            self.layers[0].b_f_updates.fill(0)
            self.layers[0].b_o_updates.fill(0)
            self.layers[0].b_c_updates.fill(0)

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

        outputs are computed as a 2000 x 50 matrix that is then flattened
        to return_buffer of 100000 records. This will be preds['train']
        """
        nrecs = inputs.shape[0]
        num_batches = int(math.floor((nrecs) / self.layers[0].nin
                                             / self.unrolls)) - 1
        outputs = self.backend.zeros((num_batches*(self.unrolls),
                                      self.batch_size))
        hidden_init = None
        cell_init = None
        for batch in xrange(num_batches):
            batch_inx = range(batch*self.layers[0].nin*self.unrolls,
                              (batch+1)*self.layers[0].nin*self.unrolls
                              + self.layers[0].nin)
            self.fprop(inputs[batch_inx, :],
                       hidden_init=hidden_init, cell_init=cell_init,
                       unrolls=self.unrolls)
            hidden_init = self.layers[0].output_list[-1]
            if 'c_t' in self.layers[0].__dict__:
                    cell_init = self.layers[0].c_t[-1]
            if (batch % self.reset_period) == 0:
                    hidden_init.fill(0)
                    if 'c_t' in self.layers[0].__dict__:
                        cell_init.fill(0)
            for tau in range(self.unrolls):
                letters = self.backend.empty(50, dtype='int32')
                self.backend.argmax(self.layers[1].output_list[tau],
                                    axis=0, out=letters)
                idx = (self.unrolls)*batch + tau
                outputs[idx, :] = letters

        return_buffer = self.backend.zeros(((num_batches+1)*self.unrolls,
                                            self.batch_size))
        return_buffer[0:num_batches*self.unrolls, :] = outputs
        return_buffer = return_buffer.transpose().reshape((-1,))
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
        items = []
        if train:
            items.append('train')
        if test:
            items.append('test')
        if validation:
            items.append('validation')

        nin = self.layers[0].nin
        targets = ds.get_inputs(train=True, test=True, validation=False)
        targets['train'] = targets['train'][nin::, :]
        targets['test'] = targets['test'][nin::, :]
        self.result = ds.backend.empty((1, 1))
        for item in items:
            if item in targets and item in preds:
                num_batches = targets[item].shape[0] / nin
                misclass = ds.backend.zeros(num_batches * nin)
                tempbuf = self.backend.zeros((num_batches + 1,
                                              self.batch_size), dtype='int32')
                for i in range(num_batches):
                    ds.backend.argmax(targets[item][i * nin:(i + 1) * nin, :],
                                      axis=0, out=tempbuf[i, :])
                import numpy as np
                misclass = tempbuf.transpose().reshape((-1,))
                tmp = misclass[6000:6018].asnumpyarray().astype(np.int8).T
                logging.info("the target for %s is %s", item,
                             tmp.view('c'))
                tmp = preds[item][6000:6018].asnumpyarray().astype(np.int8).T
                logging.info("prediction for %s is %s", item,
                             tmp.view('c'))
                ds.backend.not_equal(preds[item], misclass, misclass)
                ds.backend.mean(misclass, axes=None, out=self.result)
                logging.info("%s set misclass rate: %0.5f%%", item,
                             100 * self.result.asnumpyarray())
        # TODO: return values instead?
        if self.make_plots:
            trace()  # just used to keep figures open
