# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train distributed Google Brain models and run inference.
"""

import logging
import os
import time

from neon.models.gb import GB
from neon.layers.deprecated.layer import LocalFilteringLayerDist, LCNLayerDist
from neon.layers.deprecated.layer import L2PoolingLayerDist, LayerDist
from neon.util.compat import range
from neon.util.distarray.global_array import GlobalArray

logger = logging.getLogger(__name__)


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
        for i in range(1, self.nlayers):
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
            elif isinstance(layer, LayerDist):
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
                (self.layers[-1].nout, self.batch_size))

    def fit(self, datasets):
        inputs = datasets.get_inputs(train=True)['train']
        self.nlayers = len(self.layers)
        self.trainable_layers = []
        for ind in range(self.nlayers):
            layer = self.layers[ind]
            if isinstance(layer, LocalFilteringLayerDist):
                self.trainable_layers.append(ind)

        targets = datasets.get_targets(train=True)['train']

        # For MPI
        self.adjust_for_dist()

        if self.pretraining:
            self.pretrain(inputs, datasets)
            if self.visualize:
                self.compute_optimal_stimulus()
        if self.spot_check:
            test_inputs = datasets.get_inputs(test=True)['test']
            test_targets = datasets.get_targets(test=True)['test']
            self.check_predictions(inputs, targets, test_inputs, test_targets)
        if self.num_epochs > 0:
            self.train(inputs, targets, datasets)

    def pretrain(self, inputs, ds):
        start_time = time.time()
        logger.info('commencing unsupervised pretraining')
        num_batches = len(inputs)
        for ind in range(len(self.trainable_layers)):
            layer = self.layers[self.trainable_layers[ind]]
            pooling = self.layers[self.trainable_layers[ind] + 1]
            layer.pretrain_mode(pooling)
            for epoch in range(self.num_pretrain_epochs):
                tcost = 0.0
                trcost = 0.0
                tspcost = 0.0
                trcost_sum = 0.0
                tspcost_sum = 0.0
                for batch in range(num_batches):
                    if self.backend.mpi_rank == 0:
                        logger.debug('batch = %d', batch)
                    inputs_batch = ds.get_batch(inputs, batch)
                    output = inputs_batch
                    # Forward propagate the input all the way to
                    # the layer that we are pretraining.
                    for i in range(self.trainable_layers[ind]):
                        self.layers[i].fprop(output)
                        output = self.layers[i].output
                    rcost, spcost = layer.pretrain(output,
                                                   self.pretrain_cost,
                                                   epoch)
                    trcost += rcost
                    tspcost += spcost
                # accumulate trcost and tspcost cost across all nodes
                trcost_sum = self.backend.comm.reduce(trcost,
                                                      op=self.backend.mpi.SUM,
                                                      root=0)
                tspcost_sum = self.backend.comm.reduce(tspcost,
                                                       op=self.backend.mpi.SUM,
                                                       root=0)
                # display cost to logger on root node
                if self.backend.mpi_rank == 0:
                    tcost = trcost_sum + tspcost_sum
                    logger.info('layer: %d, epoch: %d, cost: %0.2f + %0.2f ='
                                ' %0.2f', self.trainable_layers[ind], epoch,
                                trcost / num_batches,
                                tspcost / num_batches,
                                tcost / num_batches)
                if self.visualize:
                    self.save_figs(layer.nifm, layer.ifmshape,
                                   [output, layer.defilter.output],
                                   [os.path.join('recon', 'input'),
                                    os.path.join('recon', 'output')], ind)
        logger.info('Done with pretraining')
        end_time = time.time()
        if self.backend.mpi_rank == 0:
            logger.info('%d time taken: %0.2f', self.backend.mpi_rank,
                        end_time - start_time)
        # Switch the layers from pretraining to training mode.
        for layer in self.layers:
            if isinstance(layer, LocalFilteringLayerDist):
                layer.train_mode()

    def train(self, inputs, targets, ds):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing supervised training')
        tempbuf1 = self.backend.zeros((targets[0].shape[0], self.batch_size))
        tempbuf2 = self.backend.zeros((targets[0].shape[0], self.batch_size))
        self.temp = [tempbuf1, tempbuf2]
        start_time = time.time()
        num_batches = len(inputs)
        while self.epochs.complete < self.num_epochs:
            error = 0.0
            for batch in range(num_batches):
                if self.backend.mpi_rank == 0:
                    logger.debug('batch = %d', batch)
                inputs_batch = ds.get_batch(inputs, batch)
                targets_batch = ds.get_batch(targets, batch)

                self.fprop(inputs_batch)
                if self.epochs_complete < self.num_initial_epochs:
                    # only bprop on FC layers
                    self.bprop_last(targets_batch, inputs_batch)
                else:
                    # bprop through full stack
                    self.bprop(targets_batch, inputs_batch)
                if self.backend.mpi_rank == 0:
                    error += self.cost.apply_function(self.backend,
                                                      self.layers[-1].output,
                                                      targets_batch,
                                                      self.temp)
                if self.epochs_complete < self.num_initial_epochs:
                    self.update_last(self.epochs_complete)
                else:
                    self.update(self.epochs_complete)
            if self.backend.mpi_rank == 0:
                logger.info('epoch: %d, training error: %0.5f',
                            self.epochs_complete, error / num_batches)
            self.epochs_complete += 1
        end_time = time.time()
        if self.backend.mpi_rank == 0:
            logger.info('%d time taken: %0.2f', self.backend.mpi_rank,
                        end_time - start_time)

    def bprop_last(self, targets, inputs):
        # Backprop on just the last layer.
        if self.backend.mpi_rank == 0:
            # apply derivative on root node's FC layer output
            # potential todo: for large output layers might want to distribute?
            self.error = self.cost.apply_derivative(self.backend,
                                                    self.layers[-1].output,
                                                    targets, self.temp)
            self.backend.divide(self.error, targets.shape[1], out=self.error)
        # MPI: broadcast the error matrix
        self.error._tensor = self.backend.comm.bcast(self.error.asnumpyarray())
        self.layers[-1].pre_act_ = self.layers[-1].pre_act
        self.layers[-1].bprop(self.error, self.layers[-2].output)

    def bprop(self, targets, inputs):
        i = self.nlayers - 1
        lastlayer = self.layers[i]

        error = self.backend.zeros((self.batch_size, self.layers[-1].nout))
        # apply derivative on root node's FC layer output
        if self.backend.mpi_rank == 0:
            error = self.cost.apply_derivative(self.backend,
                                               lastlayer.output, targets,
                                               self.temp)
            self.backend.divide(error, targets.shape[1], out=error)
        error._tensor = self.backend.comm.bcast(error.asnumpyarray())
        # Update the output layer.
        lastlayer.pre_act_ = lastlayer.pre_act
        lastlayer.bprop(error, self.layers[i - 1].output)

        # following code is difficult to refactor:
        # 1) LCN deltas has no halos for top layer, but does for middle layers
        # 2) L2PoolingLayerDist handles input (but not deltas) halos in its
        #    bprop
        # 3) LocalFilteringLayer needs halo handling for input and deltas

        # note: that input into LCN is ignored (self.layers[i -
        # 1].output)
        i -= 1
        self.layers[i].bprop(self.layers[i + 1].deltas,
                             self.layers[i - 1].output)
        while i > 0:
            i -= 1
            # aggregate the deltas terms at halo locations
            if isinstance(self.layers[i], LCNLayerDist):
                # note: LCN will handle halos internally because it
                # uses padding in addition to halos
                self.layers[i].bprop(
                    self.layers[
                        i + 1].input.local_array.get_bprop_view(
                        self.layers[i + 1].deltas),
                    self.layers[i - 1].output)
            elif isinstance(self.layers[i], L2PoolingLayerDist):
                # LCN layer gives a bprop view for deltas already
                self.layers[i].bprop(self.layers[i + 1].deltas,
                                     self.layers[i - 1].output)
            elif isinstance(self.layers[i], LocalFilteringLayerDist):
                self.layers[i].bprop(
                    self.layers[
                        i + 1].input.local_array.get_bprop_view(
                        self.layers[i + 1].deltas),
                    self.layers[i].input.local_array.chunk)

    def predict_set(self, ds, inputs):
        num_batches = len(inputs)
        nrecs = num_batches * self.batch_size
        if self.backend.mpi_rank == 0:
            self.outputs = self.backend.zeros((self.layers[-1].nout, nrecs))
        for batch in range(num_batches):
            inputs_batch = ds.get_batch(inputs, batch)
            self.fprop(inputs_batch)
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, nrecs)
            if self.backend.mpi_rank == 0:
                self.outputs[:, start_idx:end_idx] = self.layers[-1].output

    def predict(self, datasets, train=True, test=True, validation=True):
        """
        Generate and return predictions on the given datasets.
        """
        res = []
        for dataset in datasets:
            inputs = dataset.get_inputs(train, test, validation)
            preds = dict()
            if train and 'train' in inputs:
                self.predict_set(dataset, inputs['train'])
                if self.backend.mpi_rank == 0:
                    train_shape = (1, self.outputs.shape[0])
                    preds['train'] = dataset.backend.empty(train_shape)
                    dataset.backend.argmax(self.outputs, axis=0,
                                           out=preds['train'])
            if test and 'test' in inputs:
                self.predict_set(dataset, inputs['test'])
                if self.backend.mpi_rank == 0:
                    test_shape = (1, self.outputs.shape[0])
                    preds['test'] = dataset.backend.empty(test_shape)
                    dataset.backend.argmax(self.outputs, axis=0,
                                           out=preds['test'])
            if validation and 'validation' in inputs:
                self.predict_set(dataset, inputs['validation'])
                if self.backend.mpi_rank == 0:
                    val_shape = (1, self.outputs.shape[0])
                    preds['validation'] = dataset.backend.empty(val_shape)
                    dataset.backend.argmax(self.outputs, axis=0,
                                           out=preds['validation'])
            if self.backend.mpi_rank == 0:
                if len(preds) is 0:
                    logger.error("must specify >=1 of: train, test, "
                                 "validation")
                res.append(preds)
        return res