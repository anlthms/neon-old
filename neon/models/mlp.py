"""
Simple multi-layer perceptron model.
"""

import logging
import math

from neon.models.model import Model
from neon.models.layer import LCNLayer, L2PoolingLayer
from neon.models.layer_dist import LocalFilteringLayerDist
from mpi4py import MPI

logger = logging.getLogger(__name__)


class MLP(Model):

    """
    Fully connected, feed-forward, multi-layer perceptron model
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
        nrecs = inputs.shape[0]
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        tempbuf = self.backend.zeros((self.batch_size, self.layers[-1].nout),
                                     self.temp_dtype)
        self.temp = [tempbuf, tempbuf.copy()]

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        logger.info('commencing model fitting')
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, nrecs)
                self.fprop(inputs[start_idx:end_idx])
                self.bprop(targets[start_idx:end_idx],
                           inputs[start_idx:end_idx],
                           epoch, self.momentum)
                error += self.cost.apply_function(self.backend,
                                                  self.layers[-1].output,
                                                  targets[start_idx:end_idx],
                                                  self.temp)
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))
            for layer in self.layers:
                logger.debug("%s", layer)

    def predict_set(self, inputs):
        nrecs = inputs.shape[0]
        outputs = self.backend.zeros((nrecs, self.layers[-1].nout))
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for batch in xrange(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, nrecs)
            self.fprop(inputs[start_idx:end_idx])
            outputs[start_idx:end_idx, :] = self.layers[-1].output
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
                preds['train'] = dataset.backend.argmax(outputs, axis=1)
            if test and 'test' in inputs:
                outputs = self.predict_set(inputs['test'])
                preds['test'] = dataset.backend.argmax(outputs, axis=1)
            if validation and 'validation' in inputs:
                outputs = self.predict_set(inputs['validation'])
                preds['validation'] = dataset.backend.argmax(outputs, axis=1)
            if len(preds) is 0:
                logger.error("must specify >=1 of: train, test, validation")
            res.append(preds)
        return res

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers:
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs, epoch, momentum):
        i = self.nlayers - 1
        lastlayer = self.layers[i]
        error = self.cost.apply_derivative(self.backend,
                                           lastlayer.output, targets,
                                           self.temp)
        self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                            out=error)
        # Update the output layer.
        lastlayer.bprop(error, self.layers[i - 1].output, epoch, momentum)
        while i > 1:
            i -= 1
            self.layers[i].bprop(self.layers[i + 1].berror,
                                 self.layers[i - 1].output,
                                 epoch, momentum)
        # Update the first hidden layer.
        self.layers[i - 1].bprop(self.layers[i].berror, inputs, epoch,
                                 momentum)

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
                    misclass = ds.backend.not_equal(preds[item],
                                                    ds.backend.argmax(
                                                        targets[item], axis=1))
                    err = ds.backend.mean(misclass)
                    logging.info("%s set misclass rate: %0.5f%%" % (
                        item, 100 * err))
        # TODO: return values instead?


class MLPDist(MLP):

    """
    MPI Distributed
    Fully connected, feed-forward, multi-layer perceptron model
    """

    def fprop(self, inputs):
        y = inputs
        for layer in self.layers:
            if layer.pos < self.nlayers - 1:
                self.inputs_dist[layer.pos].local_array.local_image = y
                # perform halo exchanges
                self.inputs_dist[layer.pos].local_array.send_recv_halos()
                # make consistent chunk
                self.inputs_dist[
                    layer.pos].local_array.make_local_chunk_consistent()
                layer.fprop(self.inputs_dist[layer.pos].local_array.chunk)
                y = layer.output
            else:
                layer.fprop(y)

        self.layers[-1].pre_act._tensor = MPI.COMM_WORLD.reduce(
            self.layers[-1].pre_act.raw(), op=MPI.SUM, root=0)
        if MPI.COMM_WORLD.rank == 0:
            self.layers[-1].fprop2()

        # todo fix: broadcast back the pre_act values for bprop:
        # super-suboptimal for dist implementation,
        # but a consequence of reusing the pre_act buffer for fprop and bprop
        self.layers[-1].pre_act._tensor = MPI.COMM_WORLD.bcast(
            self.layers[-1].pre_act.raw())

    def bprop(self, targets, inputs, epoch, momentum):
        i = self.nlayers - 1
        lastlayer = self.layers[i]

        error = self.backend.zeros((self.batch_size, self.layers[-1].nout))
        if MPI.COMM_WORLD.rank == 0:
            error = self.cost.apply_derivative(self.backend,
                                               lastlayer.output, targets,
                                               self.temp)
            self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                                out=error)
        error._tensor = MPI.COMM_WORLD.bcast(error.raw())
        # Update the output layer.
        lastlayer.bprop(error, self.layers[i - 1].output, epoch, momentum)

        while i > 1:
            i -= 1
            # aggregate the berror terms at halo locations
            if isinstance(self.layers[i], LCNLayer):
                # note: LCN will handle halos internally because it
                # uses padding in addition to halos
                # Top LCN connection layer is treated differently compared to
                # middle LCN connections
                # note: that input into LCN is ignored (self.layers[i -
                # 1].output)
                if i == self.nlayers - 2:
                    self.layers[i].bprop(self.layers[i + 1].berror,
                                         self.layers[i - 1].output,
                                         epoch, momentum)
                else:
                    self.layers[i].bprop(
                        self.inputs_dist[
                            i + 1].local_array.defiltering_local_image,
                        self.layers[i - 1].output,
                        epoch, momentum)
            elif isinstance(self.layers[i], L2PoolingLayer):
                self.layers[i].bprop(self.layers[i + 1].berror,
                                     self.inputs_dist[i].local_array.chunk,
                                     epoch, momentum)
                self.inputs_dist[
                    i].local_array.defiltering_chunk = self.layers[i].berror
                self.inputs_dist[
                    i].local_array.send_recv_defiltering_layer_halos()
                self.inputs_dist[
                    i].local_array.make_defiltering_layer_consistent()
            elif isinstance(self.layers[i], LocalFilteringLayerDist):
                self.layers[i].bprop(
                    self.inputs_dist[
                        i + 1].local_array.defiltering_local_image,
                    self.inputs_dist[i].local_array.chunk,
                    epoch, momentum)
                self.inputs_dist[
                    i].local_array.defiltering_chunk = self.layers[i].berror
                self.inputs_dist[
                    i].local_array.send_recv_defiltering_layer_halos()
                self.inputs_dist[
                    i].local_array.make_defiltering_layer_consistent()

        self.layers[i - 1].bprop(
            self.inputs_dist[i].local_array.defiltering_local_image,
            self.inputs_dist[i - 1].local_array.chunk,
            epoch, momentum)
