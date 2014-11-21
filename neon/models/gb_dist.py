# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train distributed Google Brain models and run inference.
"""

import logging
import math
import os
import time

from neon.models.gb import GB
from neon.models.layer import LocalFilteringLayerDist, LCNLayerDist
from neon.models.layer import L2PoolingLayerDist, LayerWithNoBiasDist
from neon.util.compat import MPI_INSTALLED
from neon.util.distarray.global_array import GlobalArray

logger = logging.getLogger(__name__)

if MPI_INSTALLED:
    from mpi4py import MPI
else:
    logger.error('mpi4py not found')


class GBDist(GB):

    """
    MPI Distributed Google Brain class
    """

    def adjust_for_dist(self):
        # MPI: call adjust_for_dist for each layer
        layer = self.layers[0]
        layer.input = GlobalArray(cur_layer=layer,
                                  h=layer.ifmshape[0],
                                  w=layer.ifmshape[1],
                                  )
        layer.adjust_for_dist()
        for i in xrange(1, self.nlayers):
            layer = self.layers[i]
            if isinstance(layer, LocalFilteringLayerDist):
                # for h,w assumes that prev layer is a LCNLayer
                layer.input = GlobalArray(cur_layer=layer,
                                          h=self.layers[
                                              i - 1].input.local_array.height,
                                          w=self.layers[
                                              i - 1].input.local_array.width,
                                          )
            elif isinstance(layer, L2PoolingLayerDist):
                layer.input = GlobalArray(cur_layer=layer,
                                          h=self.layers[i - 1].ifmshape[0] -
                                          self.layers[i - 1].fheight + 1,
                                          w=self.layers[i - 1].ifmshape[1] -
                                          self.layers[i - 1].fwidth + 1,
                                          )
            elif isinstance(layer, LCNLayerDist):
                layer.input = GlobalArray(cur_layer=layer,
                                          h=self.layers[i - 1].ifmheight -
                                          self.layers[i - 1].fheight + 1,
                                          w=self.layers[i - 1].ifmwidth -
                                          self.layers[i - 1].fwidth + 1,
                                          # this is for padding
                                          lcn_layer_flag=True,
                                          )
                top_lcn_ifmheight = layer.ifmheight
                top_lcn_ifmwidth = layer.ifmwidth
            elif isinstance(layer, LayerWithNoBiasDist):
                # fully connected layer: no halo transfers needed
                lcn = self.layers[-2]
                layer.top_left_row_output = (
                    lcn.input.local_array.top_left_row_output)
                layer.top_left_col_output = (
                    lcn.input.local_array.top_left_col_output)
                layer.global_size = top_lcn_ifmheight * top_lcn_ifmwidth
                layer.global_width = top_lcn_ifmwidth
                layer.nin = lcn.nout
                # LCN layer doesn't have ofmshape
                layer.ifmshape = self.layers[-3].ofmshape
                layer.nifm = lcn.nifm
                # params needed for compatability with convnetdist model
                layer.prev_layer = 'LCNLayerDist'
                layer.nout_ = layer.nout
            layer.adjust_for_dist()

        if self.num_epochs > 0:
            # MPI related initializations for supervised bprop
            self.agg_output = self.backend.zeros(
                self.layers[-1].output.shape, 'float32')
            self.error = self.backend.zeros(
                (self.batch_size, self.layers[-1].nout))

    def fit(self, datasets):
        inputs = datasets[0].get_inputs(train=True)['train']
        self.nrecs, self.nin = inputs.shape
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = self.nrecs
        self.trainable_layers = []
        for ind in xrange(self.nlayers):
            layer = self.layers[ind]
            if isinstance(layer, LocalFilteringLayerDist):
                self.trainable_layers.append(ind)
            # logger.info('created layer:\n\t%s' % str(layer))

        targets = datasets[0].get_targets(train=True)['train']

        # For MPI
        self.adjust_for_dist()

        if self.pretraining:
            self.pretrain(inputs)
            if self.visualize:
                self.compute_optimal_stimulus()
        if self.spot_check:
            test_inputs = datasets[0].get_inputs(test=True)['test']
            test_targets = datasets[0].get_targets(test=True)['test']
            self.check_predictions(inputs, targets, test_inputs, test_targets)
        if self.num_epochs > 0:
            self.train(inputs, targets)

    def pretrain(self, inputs):
        start_time = time.time()
        logger.info('commencing unsupervised pretraining')
        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        for ind in range(len(self.trainable_layers)):
            layer = self.layers[self.trainable_layers[ind]]
            pooling = self.layers[self.trainable_layers[ind] + 1]
            layer.pretrain_mode(pooling)
            for epoch in xrange(self.num_pretrain_epochs):
                tcost = 0.0
                trcost = 0.0
                tspcost = 0.0
                trcost_sum = 0.0
                tspcost_sum = 0.0
                for batch in xrange(num_batches):
                    if MPI.COMM_WORLD.rank == 0:
                        logger.debug('batch = %d' % (batch))
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                    output = inputs[start_idx:end_idx]
                    # Forward propagate the input all the way to
                    # the layer that we are pretraining.
                    for i in xrange(self.trainable_layers[ind]):
                        self.layers[i].fprop(output)
                        output = self.layers[i].output
                    rcost, spcost = layer.pretrain(output,
                                                   self.pretrain_cost,
                                                   epoch)
                    trcost += rcost
                    tspcost += spcost
                # accumulate trcost and tspcost cost across all nodes
                trcost_sum = MPI.COMM_WORLD.reduce(trcost,
                                                   op=MPI.SUM, root=0)
                tspcost_sum = MPI.COMM_WORLD.reduce(tspcost,
                                                    op=MPI.SUM, root=0)
                # display cost to logger on root node
                if MPI.COMM_WORLD.rank == 0:
                    tcost = trcost_sum + tspcost_sum
                    logger.info('layer: %d, epoch: %d, cost: %0.2f + %0.2f ='
                                ' %0.2f' % (self.trainable_layers[ind], epoch,
                                            trcost / num_batches, tspcost /
                                            num_batches,
                                            tcost / num_batches))
                if self.visualize:
                    self.save_figs(layer.nifm, layer.ifmshape,
                                   [output, layer.defilter.output],
                                   [os.path.join('recon', 'input'),
                                    os.path.join('recon', 'output')], ind)
        logger.info('Done with pretraining')
        end_time = time.time()
        if MPI.COMM_WORLD.rank == 0:
            logger.info('%d time taken: %0.2f' %
                        (MPI.COMM_WORLD.rank, end_time - start_time))
        # Switch the layers from pretraining to training mode.
        for layer in self.layers:
            if isinstance(layer, LocalFilteringLayerDist):
                layer.train_mode()

    def train(self, inputs, targets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing supervised training')
        tempbuf = self.backend.zeros((self.batch_size, targets.shape[1]))
        self.temp = [tempbuf, tempbuf.copy()]
        start_time = time.time()
        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                if MPI.COMM_WORLD.rank == 0:
                    logger.debug('batch = %d' % (batch))
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                self.fprop(inputs[start_idx:end_idx])
                if epoch < self.num_initial_epochs:
                    # only bprop on FC layers
                    self.bprop_last(targets[start_idx:end_idx],
                                    inputs[start_idx:end_idx],
                                    epoch)
                else:
                    # bprop through full stack
                    self.bprop(targets[start_idx:end_idx],
                               inputs[start_idx:end_idx],
                               epoch)
                if MPI.COMM_WORLD.rank == 0:
                    error += self.cost.apply_function(self.backend,
                                                      self.layers[-1].output,
                                                      targets[
                                                          start_idx:end_idx],
                                                      self.temp)
            if MPI.COMM_WORLD.rank == 0:
                logger.info('epoch: %d, training error: %0.5f' %
                            (epoch, error / num_batches))
        end_time = time.time()
        if MPI.COMM_WORLD.rank == 0:
            logger.info('%d time taken: %0.2f' %
                        (MPI.COMM_WORLD.rank, end_time - start_time))

    def bprop_last(self, targets, inputs, epoch):
        # Backprop on just the last layer.
        if MPI.COMM_WORLD.rank == 0:
            # apply derivative on root node's FC layer output
            # potential todo: for large output layers might want to distribute?
            self.error = self.cost.apply_derivative(self.backend,
                                                    self.layers[
                                                        -1].output, targets,
                                                    self.temp)
            self.backend.divide(
                self.error, self.backend.wrap(targets.shape[0]),
                out=self.error)
        # MPI: broadcast the error matrix
        self.error._tensor = MPI.COMM_WORLD.bcast(self.error.raw())
        self.layers[-1].pre_act_ = self.layers[-1].pre_act
        self.layers[-1].bprop(
            self.error, self.layers[-2].output, epoch)

    def bprop(self, targets, inputs, epoch):
        i = self.nlayers - 1
        lastlayer = self.layers[i]

        error = self.backend.zeros((self.batch_size, self.layers[-1].nout))
        # apply derivative on root node's FC layer output
        if MPI.COMM_WORLD.rank == 0:
            error = self.cost.apply_derivative(self.backend,
                                               lastlayer.output, targets,
                                               self.temp)
            self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                                out=error)
        error._tensor = MPI.COMM_WORLD.bcast(error.raw())
        # Update the output layer.
        lastlayer.pre_act_ = lastlayer.pre_act
        lastlayer.bprop(error, self.layers[i - 1].output, epoch)

        # following code is difficult to refactor:
        # 1) LCN berror has no halos for top layer, but does for middle layers
        # 2) L2PoolingLayerDist handles input (but not berror) halos in its
        #    bprop
        # 3) LocalFilteringLayer needs halo handling for input and berror

        # note: that input into LCN is ignored (self.layers[i -
        # 1].output)
        i -= 1
        self.layers[i].bprop(self.layers[i + 1].berror,
                             self.layers[i - 1].output,
                             epoch)
        while i > 0:
            i -= 1
            # aggregate the berror terms at halo locations
            if isinstance(self.layers[i], LCNLayerDist):
                # note: LCN will handle halos internally because it
                # uses padding in addition to halos
                self.layers[i].bprop(
                    self.layers[
                        i + 1].input.local_array.get_bprop_view(
                        self.layers[i + 1].berror),
                    self.layers[i - 1].output,
                    epoch)
            elif isinstance(self.layers[i], L2PoolingLayerDist):
                # LCN layer gives a bprop view for berror already
                self.layers[i].bprop(self.layers[i + 1].berror,
                                     self.layers[i - 1].output,
                                     epoch)
            elif isinstance(self.layers[i], LocalFilteringLayerDist):
                self.layers[i].bprop(
                    self.layers[
                        i + 1].input.local_array.get_bprop_view(
                        self.layers[i + 1].berror),
                    self.layers[i].input.local_array.chunk,
                    epoch)

    def predict_set(self, inputs):
        nrecs = inputs.shape[0]
        if MPI.COMM_WORLD.rank == 0:
            self.outputs = self.backend.zeros((nrecs, self.layers[-1].nout))
        num_batches = int(math.ceil((nrecs + 0.0) / self.batch_size))
        for batch in xrange(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, nrecs)
            self.fprop(inputs[start_idx:end_idx])
            if MPI.COMM_WORLD.rank == 0:
                self.outputs[start_idx:end_idx, :] = self.layers[-1].output

    def predict(self, datasets, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given datasets.
        """
        res = []
        for dataset in datasets:
            inputs = dataset.get_inputs(train, test, validation)
            preds = dict()
            if train and 'train' in inputs:
                self.predict_set(inputs['train'])
                if MPI.COMM_WORLD.rank == 0:
                    train_shape = (self.outputs.major_axis(), 1)
                    preds['train'] = dataset.backend.empty(train_shape)
                    dataset.backend.argmax(self.outputs, axis=1,
                                           out=preds['train'])
            if test and 'test' in inputs:
                self.predict_set(inputs['test'])
                if MPI.COMM_WORLD.rank == 0:
                    test_shape = (self.outputs.major_axis(), 1)
                    preds['test'] = dataset.backend.empty(test_shape)
                    dataset.backend.argmax(self.outputs, axis=1,
                                           out=preds['test'])
            if validation and 'validation' in inputs:
                self.predict_set(inputs['validation'])
                if MPI.COMM_WORLD.rank == 0:
                    val_shape = (self.outputs.major_axis(), 1)
                    preds['validation'] = dataset.backend.empty(val_shape)
                    dataset.backend.argmax(self.outputs, axis=1,
                                           out=preds['validation'])
            if MPI.COMM_WORLD.rank == 0:
                if len(preds) is 0:
                    logger.error(
                        "must specify >=1 of: train, test, validation")
                res.append(preds)

        return res
