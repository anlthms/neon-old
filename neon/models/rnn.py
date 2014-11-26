"""
Simple recurrent neural network with one hidden layer.
"""

import logging
import math

from ipdb import set_trace as trace
import matplotlib.pyplot as plt
import sys

from neon.models.model import Model
from neon.diagnostics.visualize_rnn import VisualizeRNN

logger = logging.getLogger(__name__)

# useful for tracking down overflows and NaNs:
#import numpy as np
#np.seterr(over='raise')

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
            logger.info("%s" % str(layer))
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
        targets = inputs.copy() # override!
        nrecs = inputs.shape[inputs.major_axis()]
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        if 'ada' not in self.__dict__:
            self.ada = None
        tempbuf = self.backend.alloc(self.batch_size, self.layers[-1].nout,
                                     self.temp_dtype)
        self.temp = [tempbuf, tempbuf.copy()]
        viz = VisualizeRNN()
        num_batches = int(math.floor((nrecs + 0.0) / self.batch_size
                                                   / self.unrolls))
        print "[DEBUG] Divide input", nrecs,
        print "into batches of size", self.batch_size,
        print "with", self.unrolls, "timesteps for", num_batches, "batches"
        noisyerror = self.backend.zeros(self.num_epochs*num_batches)
        logger.info('commencing model fitting')
        suberrorlist=[]
        errorlist=[]

        for epoch in xrange(self.num_epochs):
            error = 0
            suberror = self.backend.zeros(num_batches)
            batch_inx = self.backend.zeros((self.batch_size,
                                            self.unrolls+1), dtype=int)
            hidden_init = self.backend.zeros((self.batch_size,
                                             self.layers[1].nin))
            for batch in xrange(num_batches):
                self.serve_batch(batch, batch_inx, num_batches) # get indices
                self.fprop(inputs, batch_inx, hidden_init, debug = batch==-1)
                self.bprop(targets, inputs, batch_inx, epoch, debug = batch==-1)
                hidden_init = self.layers[0].output_list[-1]
                if batch % 20 is 0: # reset hidden state periodically
                    hidden_init = self.backend.zeros((self.batch_size,
                                                     self.layers[1].nin))
                suberror = self.cost.apply_function(
                    self.backend, self.layers[-1].output_list[-1],
                    targets[batch_inx[:,-1]],
                    self.temp) / float(self.batch_size * self.layers[0].nin)
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
        
        # prediction
        inputs = datasets[0].get_inputs(test=True)['test']
        targets = datasets[0].get_targets(test=True)['test']
        batch_inx = self.backend.zeros((self.batch_size, self.unrolls+1), 
                                       dtype=int)        
        for i in range(self.unrolls+1):
            batch_inx[:, i] = self.backend.array(range(i,i+self.batch_size), 
                                                 dtype=int)
        hidden_init = self.backend.zeros((self.batch_size, self.layers[1].nin))
        self.fprop(inputs, batch_inx, hidden_init)
        
        viz.print_text(inputs[self.unrolls : self.unrolls + 50,:], 
                       self.layers[1].output_list[4])
        trace() # set a trace to prevent exiting
        # --------------------------------------------
    
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
        it row by row. 
        Wtih batch size 50 and 3 unrolls, constructing 1000 batches takes 150k data points.

        Inputs:
            batch: batch number 
            batch_inx: buffer passed for index allocation

        Returns:
            updates the buffer batch_inx directly to minimize allocations.
        """
        
        for tau in range(self.unrolls+1):
            batch_inx[:, tau] = self.unrolls*batch + tau+num_batches* self.backend.tensor_cls(range(self.batch_size))

        # was np.arange(3,5*10,10) == 3+10*np.arange(5)
        #for i in range(nlayers+1):
        #    batch_inx[:, i] = self.nlayers*batch + np.arange(i, self.batch_size*self.num_batches, self.num_batches, dtype=np.int)

    def predict_set(self, inputs):
        nrecs = inputs.shape[inputs.major_axis()]
        outputs = self.backend.alloc(nrecs, self.layers[-1].nout)
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
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
                preds['train'] = dataset.backend.argmax(
                    outputs, axis=outputs.minor_axis())
            if test and 'test' in inputs:
                outputs = self.predict_set(inputs['test'])
                preds['test'] = dataset.backend.argmax(
                    outputs, axis=outputs.minor_axis())
            if validation and 'validation' in inputs:
                outputs = self.predict_set(inputs['validation'])
                preds['validation'] = dataset.backend.argmax(
                    outputs, axis=outputs.minor_axis())
            if len(preds) is 0:
                logger.error("must specify >=1 of: train, test, validation")
            res.append(preds)
        return res

    def fprop(self, inputs, batch_inx, hidden_init, debug=False):
        """
        need to think about how to structure the fprop: 
        have a pre_act and output for every unrolling step now. The layer needs
        to keep track of all of these, so tell it which unroll we are in. 
        """
        if debug:
            import numpy as np
            print "in fprop, input", np.nonzero(inputs[0:self.unrolls, :].raw())[1]
        y = hidden_init
        for tau in range(0, self.unrolls):
            #print "FPROP unrolling step tau", tau, "of", self.unrolls
            import numpy as np
            #print "input", y[0,0], "and", np.argmax(inputs[batch_inx[:,tau], :][0,:].raw())
            self.layers[0].fprop(y, inputs[batch_inx[:,tau], :], tau) # recurrent layer
            y = self.layers[0].output_list[tau]
            #print "results in y[0,0]=", y[0,0]
            #print "FPROP output layer" # y is not cool
            self.layers[1].fprop(y, tau) # output layer
            #y = self.layers[1].output # this is an o not a y

    def bprop(self, targets, inputs, batch_inx, epoch, debug=False):
        
        full_unroll = True
        if full_unroll:
            min_unroll = 1
        else:
            min_unroll = self.unrolls

        # clear deltas 
        for tau in range(min_unroll, self.unrolls+1):
            self.layers[0].deltas[tau] *= 0
            self.layers[1].deltas_o[tau] *= 0
        
        # fill deltas 
        for tau in range(min_unroll, self.unrolls+1): 
            if debug:
                import numpy as np
                print "unrolling", tau, "of", self.unrolls
                print "in bprop, input", np.nonzero(inputs[0:tau,:].raw())[1]
                print "backprop target", np.flatnonzero(targets[tau,:].raw())
            error = self.cost.apply_derivative(self.backend, 
                                              self.layers[1].output_list[tau-1], 
                                              targets[batch_inx[:,tau]],
                                              self.temp)
            error /= float(error.shape[0] * error.shape[1])
            self.layers[1].bprop(error, 
                                 self.layers[0].output_list[tau - 1], 
                                 tau)

        cerror = self.backend.zeros((self.batch_size,self.layers[0].nout))
        for tau in range(min_unroll, self.unrolls+1): 
            
            self.backend.bprop_fc_dot(self.layers[1].deltas_o[tau], 
                                      self.layers[1].weights, out=cerror) 
            self.layers[0].bprop(cerror, inputs, tau, batch_inx)

        # apply updates
        self.layers[1].update(epoch)
        self.layers[0].update(epoch)


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
                    misclass = ds.backend.not_equal(
                        preds[item],
                        ds.backend.argmax(
                            targets[item],
                            axis=targets[item].minor_axis()))
                    err = ds.backend.mean(misclass)
                    logging.info("%s set misclass rate: %0.5f%%" % (
                        item, 100 * err))
        # TODO: return values instead?
