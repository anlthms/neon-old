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
                #trace()
                # should do a .take(indices, axis) here to get the batch out. 
                # inputs is a CPUTensor so it's all on the "device" and it's cool to index into it. 
                self.serve_batch(batch, batch_inx, num_batches)
                self.fprop(inputs,batch_inx, hidden_init)
                # --- up to here ---
                self.bprop(targets.get_minor_slice(start_idx, end_idx),
                           inputs.get_minor_slice(start_idx, end_idx),
                           epoch)
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
        
        for i in range(self.unrolls+1):
            batch_inx[:, i] = self.unrolls*batch + i+num_batches* self.backend.tensor_cls(range(self.batch_size))

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
        updates self.layer
        """
        if 0: # old
            y = inputs # 10000 x 128 how to stack for unrolling? Do an nd tensor, or make a list?
            for layer in self.layers:
                layer.fprop(y) # this modifies 
                y = layer.output

        #results_h == layers[0].pre_act and layers[0].output
        #results_o == layers[1].pre_act and layers[1].output
        
        y = hidden_init
        for unroll in range(0, self.unrolls):
            self.layers[0].fprop(y, inputs[batch_inx[:,unroll], :])
            y = self.layers[0].output
            self.layers[1].fprop(y)
            y = self.layers[1].output


        
#--------------------

    def bprop(self, targets, inputs, epoch):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        error = self.cost.apply_derivative(self.backend,
                                           lastlayer.output, targets,
                                           self.temp)
        self.backend.divide(error,
                            self.backend.wrap(targets.shape[
                                              targets.major_axis()]),
                            out=error)
        # Update the output layer.
        lastlayer.bprop(error, self.layers[i - 1].output, epoch)
        while i > 1:
            i -= 1
            self.layers[i].bprop(self.layers[i + 1].berror,
                                 self.layers[i - 1].output,
                                 epoch)
        # Update the first hidden layer.
        self.layers[i - 1].bprop(self.layers[i].berror, inputs, epoch)

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
