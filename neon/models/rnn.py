"""
Simple recurrent neural network with one hidden layer.
"""

import logging
import math

from ipdb import set_trace as trace
import matplotlib.pyplot as plt

from neon.models.model import Model

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
            logger.info("%s" % str(layer))
        inputs = datasets[0].get_inputs(train=True)['train']
        targets = datasets[0].get_targets(train=True)['train']
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

        error = []
        num_batches = int(math.floor((nrecs + 0.0) / self.batch_size))
        noisyerror = self.backend.zeros(self.num_epochs*num_batches) # not nice to dynamically allocate so use zeros.
        logger.info('commencing model fitting')
        for epoch in xrange(self.num_epochs):
            suberror=self.backend.zeros(num_batches)
            batch_inx = self.backend.zeros((self.batch_size, self.unrolls+1), dtype=int) 
            hidden_init = self.backend.zeros((self.batch_size, self.layers[1].nin))
            for batch in xrange(num_batches):
                #start_idx = batch * self.batch_size
                #end_idx = min((batch + 1) * self.batch_size, nrecs)
                # should do a .take(indices, axis) here to get the batch out. 
                # inputs is a CPUTensor so it's all on the "device" and it's cool to index into it. 
                print "[DEBUG] fit calls serve_batch"
                self.serve_batch(batch, batch_inx, num_batches)
                print "[DEBUG] fit calls fprop"
                self.fprop(inputs,batch_inx, hidden_init) # overwrites layers[].pre_act with g'
                # --- up to here ---
                print "[DEBUG] fit calls bprop"
                self.bprop(targets, inputs, batch_inx, epoch)
                print "[DEBUG] fit hidden init"
                hidden_init = something
                print "[DEBUG] fit apply cost"
                error += self.cost.apply_function(
                    self.backend, self.layers[-1].output,
                    targets.get_minor_slice(start_idx, end_idx),
                    self.temp)
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))
            for layer in self.layers:
                logger.debug("%s", layer)
    
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

        Inputs:
            batch: batch number 
            batch_inx: buffer passed for index allocation

        Returns:
            updates the buffer batch_inx directly to minimize allocations.
        """
        
        for tau in range(self.unrolls+1):
            batch_inx[:, tau] = self.unrolls*batch + tau+num_batches* self.backend.tensor_cls(range(self.batch_size))

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

    def fprop(self, inputs,batch_inx, hidden_init):
        """
        need to think about how to structure the fprop: 
        have a pre_act and output for every unrolling step now. The layer needs
        to keep track of all of these, so tell it which unroll we are in. 
        """

        #results_h == layers[0].pre_act and layers[0].output
        #results_o == layers[1].pre_act and layers[1].output
        
        y = hidden_init
        for tau in range(0, self.unrolls):
            print "unrolling step tau", tau, "of", self.unrolls
            self.layers[0].fprop(y, inputs[batch_inx[:,tau], :], tau) # recurrent layer
            y = self.layers[0].output_list[tau]
            print "output layer" # y is not cool
            self.layers[1].fprop(y, tau) # output layer
            #y = self.layers[1].output # this is an o not a y


        
#--------------------

    def bprop(self, targets, inputs, batch_inx, epoch, debug=1):
        
        # if 0: # OLD
        #     # inside each layer.bprop call, 
        #     i = self.nlayers - 1
        #     lastlayer = self.layers[i]
        #     error = self.cost.apply_derivative(self.backend,lastlayer.output, targets,self.temp)
        #     self.backend.divide(error,self.backend.wrap(targets.shape[targets.major_axis()]),out=error)
        #     # Update the output layer.
        #     lastlayer.bprop(error, self.layers[i - 1].output, epoch)
        #     # update the middel layers
        #     while i > 1:
        #         i -= 1
        #         self.layers[i].bprop(self.layers[i + 1].berror,self.layers[i - 1].output,epoch)
        #     # Update the first hidden layer.
        #     self.layers[i - 1].bprop(self.layers[i].berror, inputs, epoch)

        # NEW CODE 
        # CAN WE REFACTOR THIS SO THAT 
        updates = dict()
        updates['in']=self.backend.alloc(self.layers[0].nout, self.layers[0].nin) #64, 128
        updates['rec']=self.backend.alloc(self.layers[0].nout, self.layers[0].nout)  #64, 64
        updates['out']=self.backend.alloc(self.layers[1].nout, self.layers[1].nin) #128, 64
        temp_out=self.backend.alloc(self.layers[1].nout, self.layers[1].nin)
        temp_in=self.backend.alloc(self.layers[0].nout, self.layers[0].nin)
        temp_rec=self.backend.alloc(self.layers[0].nout, self.layers[0].nout)

        full_unroll = True # Need to set False for num_grad_check
        if full_unroll:
            min_unroll = 1
        else:
            min_unroll = self.unrolls

        for rollayers in range (min_unroll, self.unrolls+1): 
            # Print some logging messages about what input and targets are
            # if debug:
            #     import numpy as np
            #     print "unrolling", rollayers, "of", self.unrolls
            #     print "in bprop, input", np.nonzero(inputs[0:rollayers,:].raw())[1]
            #     print "backprop target", np.flatnonzero(targets[rollayers,:].raw())

            # What's send to bprop: (error, inputs)
            # all this should be able to run here, then move it out ...

            # prepare list
            # 1) output bprop
            y = self.layers[1].output_list[rollayers-1] # [1].output == result_o[1]
            # ce_de*g' == (y-targets), This is ce_de taken from mlp, uses cost=transforms.cross_entropy
            error = self.cost.apply_derivative(self.backend, y, targets[batch_inx[:,rollayers]], self.temp)
            inpu = self.layers[0].output_list[rollayers - 1]
            #!!!self.layer[1].bprop(error, inpu)
            deltas = [self.backend.alloc(y.shape[0], y.shape[1]) for k in range (rollayers+1)] # default alloc float32
            deltas[0] = error * self.layers[1].pre_act_list[rollayers-1] # confirmed == y-target
            self.backend.update_fc_dot(deltas[0], inpu, out=temp_out)
            updates['out'] += temp_out

            

            berror = self.backend.alloc(self.batch_size, self.layers[1].nin)
            self.backend.bprop_fc_dot(deltas[0], self.layers[1].weights, out=berror)
            inpu = inputs[batch_inx[:,rollayers-1]]
            #!!!self.layer[0].bprop(berror, inputs)
            deltas[1] = berror * self.layers[0].pre_act_list[rollayers-1]
            self.backend.update_fc_dot(deltas[1], inpu, temp_in)
            updates['in'] += temp_in

            # 2b) more rec -- stupid reverse loop!
            for tau in range(0, rollayers - 1)[::-1]: # go one more!
                print "unrolling", tau, "of", rollayers - 1
                self.backend.bprop_fc_dot(deltas[rollayers-tau-1], self.layers[0].weights_rec, out=berror)
                inpu = (inputs[batch_inx[:,tau]], self.layers[0].output_list[tau]) # tupel
                #!!!self.layer[0].bprop(error, inputs)
                deltas[rollayers-tau] = berror * self.layers[0].pre_act_list[tau] # for layer[0].bprop
                self.backend.update_fc_dot(deltas[rollayers-tau], inpu[0], temp_in)
                updates['in'] += temp_in
                self.backend.update_fc_dot(deltas[rollayers-(tau+1)], inpu[1], temp_rec)
                updates['rec'] += temp_rec # delta is one earlier!
                #print "recurrent analytic subupdate", np.dot(y.T, deltas[tau])[10,2]





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
