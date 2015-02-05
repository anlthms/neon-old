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
from neon.util.param import req_param, opt_param
from ipdb import set_trace as trace
logger = logging.getLogger(__name__)


class RNN(Model):

    """
    Recurrent neural network. Supports LSTM and standard RNN layers.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        req_param(self, ['layers', 'batch_size'])
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
        viz = VisualizeRNN()
        num_batches = int(math.floor((nrecs + 0.0) / 128
                                                   / self.unrolls)) - 1
        logger.info('Divide input %d into batches of size %d with %d timesteps'
                    'for %d batches',
                    nrecs, self.batch_size, self.unrolls, num_batches)
        logger.info('commencing model fitting')
        suberrorlist = []
        errorlist = []
        error = self.backend.empty((1, 1))
        suberror = self.backend.empty(num_batches)
        while self.epochs_complete < self.num_epochs:
            error.fill(0)
            suberror.fill(0)
            hidden_init = None
            cell_init = None
            for batch in range(num_batches):
                batch_inx = list(range(batch*128*self.unrolls,
                                       (batch+1)*128*self.unrolls+128))
                self.fprop(inputs[batch_inx, :],
                           hidden_init=hidden_init, cell_init=cell_init,
                           debug=(True if batch == -1 else False))
                self.bprop(targets[batch_inx, :], inputs[batch_inx, :],
                           debug=(True if batch == -1 else False))
                self.update(self.epochs_complete)
                hidden_init = self.layers[0].output_list[-1]
                if 'c_t' in self.layers[0].__dict__:
                    cell_init = self.layers[0].c_t[-1]
                if (batch % self.reset_period) == 0:  # reset hidden state
                    hidden_init.fill(0)
                    if 'c_t' in self.layers[0].__dict__:
                        cell_init.fill(0)
                self.cost.set_outputbuf(self.layers[-1].output_list[-1])
                target_out = targets[batch_inx, :][(self.unrolls-0)*128:
                                                   (self.unrolls+1)*128, :]
                suberror = self.cost.apply_function(target_out)
                self.backend.divide(suberror, float(self.batch_size *
                                                    self.layers[0].nin),
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
                        self.epochs_complete, error.asnumpyarray())
            for layer in self.layers:
                logger.debug("%s", layer)
            self.epochs_complete += 1

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
        # use targets = inputs for sequence prediction
        targets = self.backend.copy(inputs)
        nrecs = inputs.shape[0]  # was shape[1], moved to new dataset format
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        batch = 0
        batch_inx = list(range(batch*128*self.unrolls,
                               (batch+1)*128*self.unrolls+128))
        target_out = targets[batch_inx, :][(self.unrolls-0)*128:
                                           (self.unrolls+1)*128, :]

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
                float(self.batch_size * self.layers[0].nin)
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

    def fprop_eps(self, inputs, eps_tau, eps, hidden_init=None, cell_init=None,
                  debug=False, unrolls=None, num_target=None, num_i=0,
                  num_j=0):
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

    def fprop(self, inputs, hidden_init=None, cell_init=None, debug=False,
              unrolls=None):
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
        Backpropagation for a RNN.

        Notes:
            * Refactor: This bprop has an OUTER FOR LOOP over t-BPTT unrollings
              for a given unrolling depth, we go output-hidden-hidden-input
              which breaks down as: layers[1].bprop -- output layer
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
            error_h = self.layers[1].deltas
            error_c = self.backend.zeros((self.layers[1].nin,
                                          self.batch_size))
            for t in list(range(0, tau))[::-1]:  # restored to 0 as in old RNN
                self.layers[0].bprop(error_h, error_c, inputs, tau, t, numgrad)
                error_h = self.backend.copy(self.layers[0].deltas)
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
        num_batches = int(math.floor((nrecs) / 128
                                             / self.unrolls)) - 1
        outputs = self.backend.zeros((num_batches*(self.unrolls),
                                      self.batch_size))
        hidden_init = None
        cell_init = None
        for batch in range(num_batches):
            batch_inx = range(batch*128*self.unrolls,
                              (batch+1)*128*self.unrolls+128)
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

    def error_metrics(self, ds, preds,
                      train=True, test=True, validation=False):
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

    def predict_and_error(self, dataset):
        predictions = self.predict()
        self.error_metrics(dataset, predictions)


class RNNB(Model):
    """
    RNNB is the layer2 version of the RNN.
    """
    def __init__(self, **kwargs):
        self.accumulate = True
        self.dist_mode = None
        self.__dict__.update(kwargs)
        req_param(self, ['layers', 'batch_size'])
        req_param(self, ['unrolls'])
        opt_param(self, ['step_print'], -1)
        opt_param(self, ['accumulate'], False)
        self.result = 0
        kwargs = {"backend": self.backend, "batch_size": self.batch_size,
                  "accumulate": self.accumulate}
        self.data_layer = self.layers[0]
        self.rec_layer = self.layers[1]
        self.class_layer = self.layers[-2]
        self.cost_layer = self.layers[-1]
        self.link_and_initialize(self.layers, kwargs)

    def link_and_initialize(self, layer_list, kwargs, initlayer=None):
        """Copy and paste from MLPB for ease of debugging """

        for ll, pl in zip(layer_list, [initlayer] + layer_list[:-1]):
            ll.set_previous_layer(pl)
            ll.initialize(kwargs)

    # C&P from MLPB
    def print_layers(self, debug=False):
        printfunc = logger.debug if debug else logger.info
        for layer in self.layers:
            printfunc("%s", str(layer))

    def fit(self, dataset):
        viz = VisualizeRNN()
        error = self.backend.empty((1, 1))
        mb_id = self.backend.empty((1, 1))
        self.print_layers()
        self.data_layer.init_dataset(dataset)
        self.data_layer.use_set('train')
        # "output":"input":"rec"
        #           "lstm_x":"lstm_ih":"lstm_fh":"lstm_oh":"lstm_ch"
        self.grad_checker(numgrad="output")
        logger.info('commencing model fitting')
        errorlist = []
        suberrorlist = []
        suberror = self.backend.zeros((1, 1))
        while self.epochs_complete < self.num_epochs:
            error.fill(0.0)
            mb_id = 1
            self.data_layer.reset_counter()
            while self.data_layer.has_more_data():
                self.reset(mb_id)
                self.fprop(debug=(True if (mb_id is -1) else False))
                self.bprop(debug=(True if (mb_id is -1) else False))
                self.update(self.epochs_complete)

                self.cost_layer.cost.set_outputbuf(
                    self.class_layer.output_list[-1])
                suberror = self.cost_layer.get_cost()
                suberrorlist.append(float(suberror.asnumpyarray()))
                self.backend.add(error, suberror, error)
                if self.step_print > 0 and mb_id % self.step_print == 0:
                    logger.info('%d.%d logloss=%0.5f', self.epochs_complete,
                                mb_id / self.step_print - 1,
                                float(error.asnumpyarray()) /
                                self.data_layer.num_batches)
                mb_id += 1
            self.epochs_complete += 1
            errorlist.append(float(error.asnumpyarray()) /
                             self.data_layer.num_batches)
            # self.print_layers(debug=True)
            logger.info('epoch: %d, total training error: %0.5f',
                        self.epochs_complete, float(error.asnumpyarray()) /
                        self.data_layer.num_batches)
            if self.make_plots is True:
                self.plot_layers(viz, suberrorlist, errorlist)

        self.data_layer.cleanup()

    def reset(self, batch):
        """
        instead of having a separate buffer for hidden_init, we are now
        using the last element output_list[-1] for that.
        The shuffle is no longer necesseary because fprop directly looks
        into the output_list buffer.
        """
        if (batch % self.reset_period) == 0 or batch == 1:
            self.rec_layer.output_list[-1].fill(0)  # reset fprop state
<<<<<<< HEAD
            self.rec_layer.deltas.fill(0)  # reset bprop (for non-truncated)
=======
            self.rec_layer.deltas.fill(0)  # reset bprop state
>>>>>>> cccfb82b6dd5bd6142b6e702b4dc5549a542eb5b
            if 'c_t' in self.rec_layer.__dict__:
                self.rec_layer.c_t[-1].fill(0)
                self.rec_layer.celtas.fill(0)

    def plot_layers(self, viz, suberrorlist, errorlist):
        # generic error plot
        viz.plot_error(suberrorlist, errorlist)

        # LSTM specific plots
        if 'c_t' in self.rec_layer.__dict__:
            viz.plot_lstm_wts(self.rec_layer, scale=1.1, fig=4)
            viz.plot_lstm_acts(self.rec_layer, scale=21, fig=5)
        # RNN specific plots
        else:
            viz.plot_weights(self.rec_layer.weights.asnumpyarray(),
                             self.rec_layer.weights_rec.asnumpyarray(),
                             self.class_layer.weights.asnumpyarray())
            viz.plot_activations(self.rec_layer.pre_act_list,
                                 self.rec_layer.output_list,
                                 self.class_layer.pre_act_list,
                                 self.class_layer.output_list,
                                 self.cost_layer.targets)

    def fprop(self, debug=False, eps_tau=-1, eps=0,
              num_target=None, num_i=0, num_j=0):
        """
        Fixed mystery bug: Needed the _previous_ y, not the _current_ one!
        Adding numerical gradient functionality here to avoid duplicate fprops.
        TODO: Make a version where the for tau loop is inside the layer. The
        best way is to have a baseclass for both RNN and LSTM for this.
        """
        self.data_layer.fprop(None)  # get next mini batch
        inputs = self.data_layer.output
        y = self.rec_layer.output_list  # note: just a shorthand, no copy.
        c = [None for k in range(len(y))]
        if 'c_t' in self.rec_layer.__dict__:
            c = self.rec_layer.c_t

        # loop for rec_layer
        for tau in range(0, self.unrolls):
            if tau == eps_tau:
                numpy_target = num_target[num_i, num_j].asnumpyarray()
                num_target[num_i, num_j] = (numpy_target + eps)
            if debug:
                logger.debug("in RNNB.fprop, tau %d, input %d" % (tau,
                             inputs[tau].asnumpyarray().argmax(0)[0]))
            self.rec_layer.fprop(y[tau-1], c[tau-1], inputs[tau], tau)
            if tau == eps_tau:
                num_target[num_i, num_j] = numpy_target

        # loop for class_layer
        for tau in range(0, self.unrolls):
            if tau == eps_tau:
                numpy_target = num_target[num_i, num_j].asnumpyarray()
                num_target[num_i, num_j] = (numpy_target + eps)
            if debug:
                logger.debug("in RNNB.fprop, tau %d, input %d" % (tau,
                             inputs[tau].asnumpyarray().argmax(0)[0]))
            self.class_layer.fprop(y[tau], tau)
            if tau == eps_tau:
                num_target[num_i, num_j] = numpy_target
        # cost layer fprop is a pass.

    def bprop(self, debug, numgrad=None):
        """
        Parent method for bptt and truncated-bptt. Truncation is neccessary
        for the standard RNN as a way to prevent exploding gradients. For the
        LSTM it also
        """
        if self.truncate:
            self.trunc_bprop_tt(debug, numgrad)
        else:
            self.bprop_tt(debug, numgrad)

    def trunc_bprop_tt(self, debug, numgrad=None):
        """
        TODO: move the loop over t into the layer class.
        """
        if numgrad is None:
            min_unroll = 1
        else:
            logger.debug("MLPB.bprop single unrolling for numgrad")
            min_unroll = self.unrolls

        for tau in range(min_unroll-0, self.unrolls+1):
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[tau-1])
            self.cost_layer.bprop(None, tau-1)
            if debug:
                tmp = self.cost_layer.targets[tau-1].asnumpyarray()
                tmp = tmp.argmax(0)[0]
                logger.debug("in RNNB.bprop, tau %d target %d" % (tau-1, tmp))
            error = self.cost_layer.deltas
            self.class_layer.bprop(error, tau, numgrad=numgrad)
<<<<<<< HEAD
            error = self.class_layer.deltas
=======
            # OLD: top level bprop gets only ouput layer delts
            error = self.backend.zeros(self.class_layer.deltas.shape)
            error[:] = self.class_layer.deltas
            # NEW: Mixing in errors from hidden and output layer
            self.backend.add(self.class_layer.deltas, self.rec_layer.deltas,
                             out=error)  # mix in state!
            # NICE: With this addition, GRADPLOSION iminent!
>>>>>>> cccfb82b6dd5bd6142b6e702b4dc5549a542eb5b
            for t in list(range(0, tau))[::-1]:
                if 'c_t' in self.rec_layer.__dict__:
                    cerror = self.rec_layer.celtas  # on t=0, prev batch state
                else:
                    cerror = None  # for normal RNN
                self.rec_layer.bprop(error, cerror, t, numgrad=numgrad)
                error[:] = self.rec_layer.deltas  # [TODO] why need deepcopy?
<<<<<<< HEAD

    def bprop_tt(self, debug, numgrad=None):
        """
        Keep state over consecutive unrollings. Explodes for RNN, and is not
        currently used for anything, but future recurrent layers might use it.
        """

        temp1 = self.backend.zeros(self.class_layer.deltas.shape)
        temp2 = self.backend.zeros(self.class_layer.deltas.shape)
        temp1c = self.backend.zeros(self.class_layer.deltas.shape)
        temp2c = self.backend.zeros(self.class_layer.deltas.shape)

        for tau in list(range(self.unrolls))[::-1]:
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[tau])
            self.cost_layer.bprop(None, tau)
            cost_error = self.cost_layer.deltas
            self.class_layer.bprop(cost_error, tau, numgrad=numgrad)

            external_error = self.class_layer.deltas
            internal_error = self.rec_layer.deltas
            if 'c_t' in self.rec_layer.__dict__:
                internal_cerror = self.rec_layer.celtas
                external_cerror = self.backend.zeros(temp1.shape)
            else:
                internal_cerror = None
                external_cerror = None

            self.rec_layer.bprop(external_error, external_cerror, tau,
                                 numgrad=numgrad)
            temp1[:] = self.rec_layer.deltas
            if 'c_t' in self.rec_layer.__dict__:
                temp1c[:] = self.rec_layer.celtas
            self.rec_layer.bprop(internal_error, internal_cerror, tau,
                                 numgrad=numgrad)
            temp2[:] = self.rec_layer.deltas
            if 'c_t' in self.rec_layer.__dict__:
                temp2c[:] = self.rec_layer.celtas
            self.backend.add(temp1, temp2, out=self.rec_layer.deltas)
            if 'c_t' in self.rec_layer.__dict__:
                self.backend.add(temp1c, temp2c, out=self.rec_layer.celtas)
=======
>>>>>>> cccfb82b6dd5bd6142b6e702b4dc5549a542eb5b

    def update(self, epoch):
        '''straight from old RNN == MLP == MLPB'''
        for layer in self.layers:
            layer.update(epoch)  # update also zeros out update buffers.

    # taken from RNN, really need this.
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

        if numgrad is "output":
            num_target = self.class_layer.weights
            anl_target = self.class_layer.weight_updates
            num_i, num_j = 15, 56
        elif numgrad is "input":
            num_target = self.rec_layer.weights
            anl_target = self.rec_layer.weight_updates
            num_i, num_j = 12, 110  # 110 is "n"
        elif numgrad is "rec":
            num_target = self.rec_layer.weights_rec
            anl_target = self.rec_layer.updates_rec
            num_i, num_j = 12, 63
        elif numgrad is "lstm_x":
            num_target = self.rec_layer.Wfx
            anl_target = self.rec_layer.Wfx_updates
            num_i, num_j = 12, 110
        elif numgrad is "lstm_ih":
            num_target = self.rec_layer.Wih
            anl_target = self.rec_layer.Wih_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_fh":
            num_target = self.rec_layer.Wfh
            anl_target = self.rec_layer.Wfh_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_oh":
            num_target = self.rec_layer.Woh
            anl_target = self.rec_layer.Woh_updates
            num_i, num_j = 12, 55
        elif numgrad is "lstm_ch":
            num_target = self.rec_layer.Wch
            anl_target = self.rec_layer.Wch_updates
            num_i, num_j = 12, 55

        eps = 1e-6  # better to use float64 in cpu.py for this
        numerical = 0  # initialize buffer
        #  loop to inject epsilon in different unrolling stages
        for eps_tau in range(0, self.unrolls):
            self.reset(1)  # clear hidden input
            self.fprop(debug=False, eps_tau=eps_tau, eps=0,
                       num_target=num_target, num_i=num_i, num_j=num_j)
            self.cost_layer.set_targets()
            self.data_layer.reset_counter()
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[-1])
            suberror_eps = self.cost_layer.get_cost().asnumpyarray()

            self.reset(1)
            self.fprop(debug=False, eps_tau=eps_tau, eps=eps,
                       num_target=num_target, num_i=num_i, num_j=num_j)
            self.data_layer.reset_counter()
            self.cost_layer.cost.set_outputbuf(
                self.class_layer.output_list[-1])
            suberror_ref = self.cost_layer.get_cost().asnumpyarray()

            num_part = (suberror_eps - suberror_ref) / eps
            logger.debug("numpart for  eps_tau=%d of %d is %e",
                         eps_tau, self.unrolls, num_part)
            numerical += num_part

        # bprop for analytical gradient
        self.bprop(debug=False, numgrad=numgrad)

        analytical = anl_target[num_i, num_j].asnumpyarray()
        logger.debug("---------------------------------------------")
        logger.debug("RNN grad_checker: suberror_eps %f", suberror_eps)
        logger.debug("RNN grad_checker: suberror_ref %f", suberror_ref)
        logger.debug("RNN grad_checker: numerical %e", numerical)
        logger.debug("RNN grad_checker: analytical %e", analytical)
        logger.debug("RNN grad_checker: ratio %e", 1./(numerical/analytical))
        logger.debug("---------------------------------------------")

    # adapted from MLPB, added time unrolling
    def predict_and_error(self, dataset=None):
        """
        todo: take the
            outputs[idx, :] = letters
        stuff from predict_set and use it to descramble the predictions
        like we had before
        """
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
            outputs_pred = self.backend.zeros(
                ((self.data_layer.num_batches + 0)
                 * (self.unrolls), self.batch_size))
            outputs_targ = self.backend.zeros(
                ((self.data_layer.num_batches + 0)
                 * (self.unrolls), self.batch_size))
            mb_id = 0
            self.data_layer.reset_counter()
            while self.data_layer.has_more_data():
                mb_id += 1
                self.reset(mb_id)
                self.fprop(debug=False)
                # added time unrollig loop to disseminate fprop resuluts
                for tau in range(self.unrolls):
                    probs = self.class_layer.output_list[tau]
                    targets = self.data_layer.targets[tau]
                    self.backend.argmax(targets, axis=0, out=labels)
                    self.backend.argmax(probs, axis=0, out=predlabels)
                    self.backend.not_equal(predlabels, labels, misclass)
                    self.backend.sum(misclass, axes=None, out=batch_sum)
                    self.backend.add(misclass_sum, batch_sum, misclass_sum)
                    self.backend.sum(self.cost_layer.cost.apply_logloss(
                                     targets), axes=None, out=batch_sum)
                    self.backend.add(logloss_sum, batch_sum, logloss_sum)

                    # collect batches to re-assemble continuous data
                    idx = (self.unrolls)*(mb_id-1) + tau
                    outputs_pred[idx, :] = predlabels
                    outputs_targ[idx, :] = labels

            self.write_string(outputs_pred, outputs_targ, setname)
            self.result = misclass_sum.asnumpyarray()[0, 0] / (nrecs *
                                                               self.unrolls)
            self.data_layer.cleanup()
            return_err[setname] = self.result
            logging.info("%s set misclass rate: %0.5f%% logloss %0.5f" % (
                setname, 100 * misclass_sum.asnumpyarray() / nrecs /
                self.unrolls, logloss_sum.asnumpyarray() / nrecs /
                self.unrolls))
        if self.make_plots is True:
            trace()  # stop to look at plots
        return return_err

    def write_string(self, pred, targ, setname):
            """ For text prediction, reassemble the batches and print out a
            short contigous segment of target text and predicted text - useful
            to check for off-by-one errors and the like"""
            import numpy as np

            # flatten the predictions
            pred_flat = pred.transpose().reshape((-1,))
            pred_int = pred_flat[2:40].asnumpyarray()[:, 0].astype(np.int8).T
            targ_flat = targ.transpose().reshape((-1,))
            targ_int = targ_flat[2:40].asnumpyarray()[:, 0].astype(np.int8).T
            # remove special characters, replace them with '#'
            pred_int[pred_int < 32] = 35
            targ_int[targ_int < 32] = 35

            # create output strings
            logging.info("the target for '%s' is: '%s'", setname,
                         ''.join(targ_int.view('c')))
            logging.info("prediction for '%s' is: '%s'", setname,
                         ''.join(pred_int.view('c')))
