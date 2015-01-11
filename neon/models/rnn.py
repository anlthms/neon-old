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
    Recurrent neural network
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
        self.grad_checker()  # check gradients first

        """
        Learn model weights on the given dataset.
        """
        for layer in self.layers:
            logger.info("XXX %s", str(layer))
        inputs = dataset.get_inputs(train=True)['train']
        # append an extra zero element to account for overflow
        # inputs = self.backend.zeros((inputset.shape[0]+1, inputset.shape[1]))
        # inputs[0:inputset.shape[0], 0:inputset.shape[1]] = inputset
        # no idea how to do this for the new data format!
        targets = inputs.copy()  # use targets = inputs for sequence prediction
        nrecs = inputs.shape[0]  # was shape[1], moved to new dataset format
        viz = VisualizeRNN()
        num_batches = int(math.floor((nrecs + 0.0) / 128
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
            hidden_init = None
            cell_init = None
            for batch in xrange(num_batches):
                batch_inx = xrange(batch*128*self.unrolls,
                                   (batch+1)*128*self.unrolls+128)
                self.fprop(inputs[batch_inx, :], hidden_init=hidden_init,
                           cell_init=cell_init,
                           debug=(True if batch == -1 else False))
                self.bprop(targets[batch_inx, :], inputs[batch_inx, :],
                           debug=(True if batch == -1 else False))
                self.update(epoch)
                hidden_init = self.layers[0].output_list[-1]
                if 'c_t' in self.layers[0].__dict__:
                    cell_init = self.layers[0].c_t[-1]
                if batch % 20 is 0:  # reset hidden state periodically
                    self.backend.fill(hidden_init, 0)
                    if 'c_t' in self.layers[0].__dict__:
                        self.backend.fill(cell_init, 0)
                self.cost.set_outputbuf(self.layers[-1].output_list[-1])
                target_out = targets[batch_inx, :][(self.unrolls-0)*128:
                                                   (self.unrolls+1)*128, :]
                # print "i", self.layers[0].b_i[0:3].transpose(), "f", self.layers[0].b_f[0:3].transpose(), "o", self.layers[0].b_o[0:3].transpose(), "g", self.layers[0].b_c[0:3].transpose()
                suberror = self.cost.apply_function(target_out)
                suberror /= float(self.batch_size * self.layers[0].nin)
                suberrorlist.append(suberror)
                error += suberror / num_batches
            errorlist.append(error)
            if self.make_plots is True:
                viz.plot_weights(self.layers[0].weights.raw(),
                                 self.layers[0].Wih.raw(),
                                 self.layers[1].weights.raw())
                viz.plot_lstm(self.layers[0].Wix.raw(),
                              self.layers[0].Wfx.raw(),
                              self.layers[0].Wox.raw(),
                              self.layers[0].Wcx.raw(),
                              self.layers[0].Wih.raw(),
                              self.layers[0].Wfh.raw(),
                              self.layers[0].Woh.raw(),
                              self.layers[0].Wch.raw(),
                              fig=4)
                viz.plot_lstm(self.layers[0].i_t[0].raw(),
                              self.layers[0].f_t[0].raw(),
                              self.layers[0].o_t[0].raw(),
                              self.layers[0].g_t[0].raw(),
                              self.layers[0].net_i[0].raw(),
                              self.layers[0].net_f[0].raw(),
                              self.layers[0].net_o[0].raw(),
                              self.layers[0].net_g[0].raw(),
                              fig=5)
                viz.plot_error(suberrorlist, errorlist)
                viz.plot_activations(self.layers[0].net_i,
                                     self.layers[0].i_t,
                                     self.layers[1].pre_act_list,
                                     self.layers[1].output_list,
                                     targets[batch_inx, :])
            logger.info('epoch: %d, total training error per element: %0.5f',
                        epoch, error)
            for layer in self.layers:
                logger.debug("%s", layer)

    def grad_checker(self):
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
        targets = inputs.copy()  # use targets = inputs for sequence prediction
        nrecs = inputs.shape[0]  # was shape[1], moved to new dataset format
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        batch = 0
        batch_inx = xrange(batch*128*self.unrolls,
                           (batch+1)*128*self.unrolls+128)
        target_out = targets[batch_inx, :][(self.unrolls-0)*128:
                                           (self.unrolls+1)*128, :]
        # ----------------------------------------
        num_target = self.layers[1].weights # num
        an_target = self.layers[1].weight_updates # anal factor 4
        num_i, num_j = 12, 56 # for output

        num_target = self.layers[0].weights #  num gradient -1.085769e-04
        an_target = self.layers[0].weight_updates #  anal factor 4
        num_i, num_j = 12, 110 # for input, 110 is "n"

        #num_target = self.layers[0].weights_rec # num gradient 1.462686e-04
        #an_target = self.layers[0].updates_rec # anal fac 4 to 3.659200e-05
        #num_i, num_j = 12, 63 # for recurrentl

        #num_target = self.layers[0].Wfx
        #an_target = self.layers[0].Wfx_updates


        # ----------------------------------------
        eps = 1e-6  # use float64 in cpu.py for this
        numerical = 0 # initialize buffer
        # extra loop to inject epsilon in different unrolling stages
        for tau in range(0, self.unrolls):
            print "CALLING PFROP WITH EPSILON, tau=", tau
            self.fprop_eps(inputs[batch_inx, :], tau, eps, hidden_init=None,
                           debug=(True if batch == -1 else False),
                           num_target=num_target, num_i=num_i, num_j=num_j)
            self.cost.set_outputbuf(self.layers[-1].output_list[-1])
            #fuck_eps = self.layers[1].output_list[3]
            suberror_eps = self.cost.apply_function(target_out)

            print "CALLING PFROP WITHOUT EPSILON, tau=", tau
            self.fprop_eps(inputs[batch_inx, :], tau, 0, hidden_init=None,
                           debug=(True if batch == -1 else False),
                           num_target=num_target, num_i=num_i, num_j=num_j)
            self.cost.set_outputbuf(self.layers[-1].output_list[-1])
            #fuck_ref = self.layers[1].output_list[3]
            suberror_ref = self.cost.apply_function(target_out)
            num_part = (suberror_eps - suberror_ref) / eps / \
                float(self.batch_size * self.layers[0].nin)
            #fuck_part = fuck_ref.raw().sum() - fuck_eps.raw().sum()
            #print "diff in layers[1].output_list[3]", fuck_part
            logger.info("numpart for  tau=%d of %d is %e",
                        tau, self.unrolls, num_part)
            numerical += num_part

        # bprop for comparison
        self.bprop(targets[batch_inx, :], inputs[batch_inx, :], numgrad=True)

        analytical = an_target[num_i, num_j].raw()
        logger.info("RNN grad_checker: suberror_eps %f", suberror_eps)
        logger.info("RNN grad_checker: suberror_ref %f", suberror_ref)
        logger.info("RNN grad_checker: numerical %e", numerical)
        logger.info("RNN grad_checker: analytical %e", analytical)
        logger.info("RNN grad_checker: ratio %e", numerical/analytical)
        trace()  # off by a factor of 4 for Wout.

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
                num_target[num_i, num_j] = num_target[num_i, num_j].raw() + eps

            self.layers[0].fprop(y=y, inputs=inputs[nin*tau:nin*(tau+1), :],
                                 tau=tau, cell=c)
            y = self.layers[0].output_list[tau]
            if 'c_t' in self.layers[0].__dict__:
                c = self.layers[0].c_t[tau]
            self.layers[1].fprop(inputs=y, tau=tau)

            if tau == eps_tau:
                num_target[num_i, num_j] = num_target[num_i, num_j].raw() - eps

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
              debug=False, numgrad=False):
        """
        Refactor:
        This bprop has an OUTER FOOR LOOP over t-BPTT unrollings
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
        full_unroll = 1-numgrad  # don't unroll if num grad is done
        if full_unroll:
            min_unroll = 1
        else:
            min_unroll = self.unrolls
            print "bprop skipping partial unrolls"

        # clear updates [TODO] Move these to layer.update
        if 'weight_updates' in self.layers[0].__dict__:
            self.backend.fill(self.layers[0].weight_updates, 0)
        if 'updates_rec' in self.layers[0].__dict__:
            self.backend.fill(self.layers[0].updates_rec, 0)
        self.backend.fill(self.layers[1].weight_updates, 0)
        if 'Wix_updates' in self.layers[0].__dict__:
            # reset these things back to zero
            #for a in ['i', 'f', 'o', 'c',]:
            #    for b in ['x', 'h']:
            self.backend.fill(self.layers[0].Wix_updates, 0)
            self.backend.fill(self.layers[0].Wfx_updates, 0)
            self.backend.fill(self.layers[0].Wox_updates, 0)
            self.backend.fill(self.layers[0].Wcx_updates, 0)
            self.backend.fill(self.layers[0].Wih_updates, 0)
            self.backend.fill(self.layers[0].Wfh_updates, 0)
            self.backend.fill(self.layers[0].Woh_updates, 0)
            self.backend.fill(self.layers[0].Wch_updates, 0)

        # this loop is a property of t-BPTT through different depth.
        # inside this loop, go through the input-hidden-output stack.
        for tau in range(min_unroll, self.unrolls+1):
            if debug:
                logger.info("bprop target %d of %d is: %f", tau, self.unrolls,
                            targets[nin*tau:nin*(tau+1), :].argmax(0)[0])

            # output layers[1]:
            self.cost.set_outputbuf(self.layers[1].output_list[tau - 1])
            error = self.cost.apply_derivative(targets[nin*tau:nin*(tau+1), :])
            esize = error.shape[0] * error.shape[1]
            self.backend.divide(error, esize, out=error)
            self.layers[1].bprop(error,
                                 self.layers[0].output_list[tau - 1], tau)

            # recurrent layers[0]: loop over different unrolling sizes
            error_h = self.layers[1].berror
            error_c = self.backend.zeros((self.layers[1].nin,
                                          self.batch_size))
            for t in list(range(0, tau))[::-1]: # restored to 0 as in old RNN
                self.layers[0].bprop(error_h, error_c, inputs, tau, t, numgrad)
                error_h = self.layers[0].berror
                if 'cerror' in self.layers[0].__dict__:
                    error_c = self.layers[0].cerror
            # # last layer: put output[-1], i.e. hidden init, into output[end]
            # """Do we even need to bprop this deep? Computes an error that is
            # not used anywhere, W_h into zero, only update to W_x is used! """
            # # This is not the culprit, makes no difference.
            # if True:
            #     t = 0
            #     # assuming it's ok to overwrite? (These are reused throughout!)
            #     self.layers[0].output_list[tau - 1] = hidden_init  # TESTING
            #     if 'c_t' in self.layers[0].__dict__:
            #         self.layers[0].c_t[tau - 1] = cell_init  # TESTING
            #     self.layers[0].bprop(error_h, error_c, inputs, tau, t, numgrad)

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
        num_batches = int(math.floor((nrecs) / 128
                                             / self.unrolls)) - 1
        outputs = self.backend.zeros((num_batches*(self.unrolls),
                                      self.batch_size))
        hidden_init = self.backend.zeros((self.layers[1].nin, self.batch_size))
        for batch in xrange(num_batches):
            batch_inx = range(batch*128*self.unrolls,
                              (batch+1)*128*self.unrolls+128)
            self.fprop(inputs[batch_inx, :], hidden_init, unrolls=self.unrolls)
            hidden_init = self.layers[0].output_list[-1]
            if batch % 20 is 0:
                    hidden_init = self.backend.zeros((self.layers[1].nin,
                                                      self.batch_size))
            for tau in range(self.unrolls):
                letters = self.backend.empty(50, dtype=int)
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
        if len(preds) is 0:
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
        for item in items:
            if item in targets and item in preds:
                num_batches = targets[item].shape[0]/nin
                misclass = ds.backend.zeros(num_batches*nin)
                tempbuf = self.backend.zeros((num_batches+1,
                                              self.batch_size))
                for i in range(num_batches):
                    ds.backend.argmax(targets[item][i*nin:(i+1)*nin, :],
                                      axis=0, out=tempbuf[i, :])
                import numpy as np
                misclass = tempbuf.transpose().reshape((-1,))
                tmp = misclass[6000:6018].raw().astype(np.int8)
                logging.info("the target for %s is %s", item,
                             tmp.view('c'))
                tmp = preds[item][6000:6018].raw().astype(np.int8)
                logging.info("prediction for %s is %s", item,
                             tmp.view('c'))
                ds.backend.not_equal(preds[item], misclass, misclass)
                self.result = ds.backend.mean(misclass)
                logging.info("%s set misclass rate: %0.5f%%", item,
                             100 * self.result)
        # TODO: return values instead?
        trace()  # just used to keep figures open
