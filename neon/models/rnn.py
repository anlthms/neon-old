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
from ipdb import set_trace as trace


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
        # append an extra zero element to account for overflow
        # inputs = self.backend.zeros((inputset.shape[0]+1, inputset.shape[1]))
        # inputs[0:inputset.shape[0], 0:inputset.shape[1]] = inputset
        # no idea how to do this for the new data format!
        targets = inputs.copy()  # use targets = inputs for sequence prediction
        nrecs = inputs.shape[0]  # was shape[1], moved to new dataset format
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
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
            for batch in xrange(num_batches):
                batch_inx = xrange(batch*128*self.unrolls,
                                  (batch+1)*128*self.unrolls+128)
                self.fprop(inputs[batch_inx, :], hidden_init=hidden_init,
                           debug=(True if batch == -1 else False))
                self.bprop(targets[batch_inx, :], inputs[batch_inx, :],
                           debug=(True if batch == -1 else False))
                self.update(epoch)
                hidden_init = self.layers[0].output_list[-1]
                if batch % 20 is 0:  # reset hidden state periodically
                    hidden_init = self.backend.zeros((self.layers[1].nin,
                                                     self.batch_size))
                self.cost.set_outputbuf(self.layers[-1].output_list[-1])
                target_out = targets[batch_inx, :][(self.unrolls-0)*128:
                                                   (self.unrolls+1)*128, :]
                suberror = self.cost.apply_function(target_out)
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
                                     targets[batch_inx, :])
            logger.info('epoch: %d, total training error per element: %0.5f' %
                        (epoch, error))
            for layer in self.layers:
                logger.debug("%s", layer)


    def fprop(self, inputs, hidden_init=None, debug=False, unrolls=None):
        """
        have a pre_act and output for every unrolling step. The layer needs
        to keep track of all of these, so we tell it which unroll we are in.
        """
        if hidden_init is None:
            hidden_init = self.backend.zeros((self.layers[1].nin,
                                              self.batch_size))
        if unrolls is None:
            unrolls = self.unrolls
        if debug:
            import numpy as np
            print "fprop input"
            print inputs.reshape((6,128,50)).argmax(1)[:,0:10]
            #trace()
        y = hidden_init
        for tau in range(0, unrolls):
            self.layers[0].fprop(y, inputs[128*tau:128*(tau+1), :], tau)
            y = self.layers[0].output_list[tau]
            self.layers[1].fprop(y, tau)

    def bprop(self, targets, inputs, debug=False):

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
                print "backprop target", tau, "of", self.unrolls, "is", targets[128*tau:128*(tau+1), :].argmax(0)[0]
            self.cost.set_outputbuf(self.layers[1].output_list[tau - 1])
            error = self.cost.apply_derivative(targets[128*tau:128*(tau+1), :])
            #print "when tau", tau, "going to", 128*(tau+1)
            error /= float(error.shape[0] * error.shape[1])
            self.layers[1].bprop(error,
                                 self.layers[0].output_list[tau - 1],
                                 tau)

        cerror = self.backend.zeros((self.layers[0].nout, self.batch_size))
        for tau in range(min_unroll, self.unrolls+1):
            # need to bprop from the output layer before calling bprop
            self.backend.bprop_fc(self.layers[1].deltas_o[tau],
                                  self.layers[1].weights,
                                  out=cerror)
            self.layers[0].bprop(cerror, inputs, tau)

    def update(self, epoch):
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

        outputs are computed as a 2000 x 50 matrix that is then flattened
        to return_buffer of 100000 records. This will be preds['train']
        """
        nrecs = inputs.shape[0] # not sure what recs is, but no.
        num_batches = int(math.floor((nrecs) / 128
                                             / self.unrolls)) - 2
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
        return_buffer = self.backend.zeros(nrecs/128*50)
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
            # targets = ds.get_targets(train=True, test=True, validation=False)
            targets = ds.get_inputs(train=True, test=True, validation=False)
            targets['train'] = targets['train'][128::, :]
            targets['test'] = targets['test'][128::, :]
            for item in items:
                print "item:", item
                if item in targets and item in preds:
                    num_batches = targets[item].shape[0]/128
                    misclass = ds.backend.zeros(num_batches*128) # 255872
                    # argmax is now over a not cool direction.
                    tempbuf = self.backend.zeros((num_batches+1,50))
                    for i in range(num_batches):
                        ds.backend.argmax(targets[item][i*128:(i+1)*128, :],
                                          axis=0, out=tempbuf[i,:])
                    import numpy as np
                    misclass = tempbuf.transpose().reshape((-1,))
                    print "the target for", item, "is", misclass[00:18].raw().astype(np.int8).view('c')
                    print "prediction for", item, "is", preds[item][00:18].raw().astype(np.int8).view('c')
                    ds.backend.not_equal(preds[item], misclass, misclass)
                    self.result = ds.backend.mean(misclass)
                    logging.info("%s set misclass rate: %0.5f%%" % (
                        item, 100 * self.result))
        # TODO: return values instead?
