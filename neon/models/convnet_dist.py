# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple multi-layer perceptron model.
"""

import logging
import math

from neon.models.mlp_dist import MLPDist
from neon.models.layer import ConvLayerDist, MaxPoolingLayerDist
from neon.models.layer import LayerWithNoBiasDist
from neon.util.compat import MPI_INSTALLED
from neon.util.distarray.global_array import GlobalArray

logger = logging.getLogger(__name__)

if MPI_INSTALLED:
    from mpi4py import MPI
else:
    logger.error('mpi4py not found')


class ConvnetDist(MLPDist):

    """
    Halo/tower distributed convolutional network
    """

    def adjust_for_dist(self):
        # MPI: call adjust_for_dist for each layer
        layer = self.layers[0]
        layer.input = GlobalArray(cur_layer=layer)
        layer.adjust_for_dist()
        for i in xrange(1, self.nlayers):
            layer = self.layers[i]
            logger.debug('layer= %d', i)
            if isinstance(layer, ConvLayerDist):
                # for h,w assumes that prev layer is a LCNLayer
                layer.input = GlobalArray(cur_layer=layer,
                                          h=self.layers[i - 1].ofmheight,
                                          w=self.layers[i - 1].ofmwidth)
            elif isinstance(layer, MaxPoolingLayerDist):
                layer.input = GlobalArray(cur_layer=layer,
                                          h=self.layers[i - 1].ofmheight,
                                          w=self.layers[i - 1].ofmwidth)
                top_mp_ifmheight = layer.ifmheight
                top_mp_ifmwidth = layer.ifmwidth
            elif isinstance(layer, LayerWithNoBiasDist):
                # fully connected layer: no halo transfers needed
                layer.nout_ = layer.nout
                if i < self.nlayers - 1:
                    if layer.nout % MPI.COMM_WORLD.size != 0:
                        raise ValueError('Unsupported layer.nout % '
                                         'MPI.COMM_WORLD.size != 0')
                    layer.nout = layer.nout / MPI.COMM_WORLD.size
                if isinstance(self.layers[i - 1], MaxPoolingLayerDist):
                    mp_layer = self.layers[i - 1]
                    layer.top_left_row_output = (
                        mp_layer.input.local_array.top_left_row_output)
                    layer.top_left_col_output = (
                        mp_layer.input.local_array.top_left_col_output)
                    # global dims of the input to this layer
                    layer.global_width = (top_mp_ifmwidth - self.layers[
                                          i - 1].fwidth) / (
                        self.layers[i - 1].stride) + 1
                    layer.global_height = (top_mp_ifmheight - self.layers[
                                           i - 1].fheight) / (
                        self.layers[i - 1].stride) + 1
                    layer.global_size = (layer.global_height *
                                         layer.global_width)
                    logger.debug(
                        'global_size=%d, global_width=%d', layer.global_size,
                        layer.global_width)
                    layer.nin = mp_layer.nout
                    layer.ifmshape = mp_layer.ofmshape
                    layer.nifm = mp_layer.nifm
                    layer.prev_layer = 'MaxPoolingLayerDist'
                elif isinstance(self.layers[i - 1], LayerWithNoBiasDist):
                    # split the inputs nin across MPI.COMM_WORLD.size
                    if layer.nin % MPI.COMM_WORLD.size != 0:
                        raise ValueError('Unsupported layer.nin % '
                                         'MPI.COMM_WORLD.size != 0')
                    layer.nin = layer.nin / MPI.COMM_WORLD.size
                    layer.in_indices = range(MPI.COMM_WORLD.rank * layer.nin,
                                             (MPI.COMM_WORLD.rank + 1) *
                                             layer.nin)
                    layer.prev_layer = 'LayerWithNoBiasDist'
                else:
                    raise ValueError('Unsupported previous layer for '
                                     'LayerWithNoBiasDist')
            layer.adjust_for_dist()

        if self.num_epochs > 0:
            # MPI related initializations for supervised bprop
            self.agg_output = self.backend.zeros(
                self.layers[-1].output.shape, 'float32')
            self.error = self.backend.zeros(
                (self.layers[-1].nout, self.batch_size))

    def fit(self, datasets):
        """
        Learn model weights on the given datasets.
        """
        # for layer in self.layers:
        #    logger.info("%s" % str(layer))
        self.comm = MPI.COMM_WORLD
        self.adjust_for_dist()
        ds = datasets[0]
        inputs = ds.get_inputs(train=True)['train']
        targets = ds.get_targets(train=True)['train']
        nrecs = inputs.shape[0]
        if 'batch_size' not in self.__dict__:
            self.batch_size = nrecs
        if 'temp_dtype' not in self.__dict__:
            self.temp_dtype = None
        tempbuf = self.backend.zeros((self.layers[-1].nout, self.batch_size),
                                     self.temp_dtype)
        self.temp = [tempbuf, tempbuf.copy()]

        # we may include 1 smaller-sized partial batch if num recs is not an
        # exact multiple of batch size.
        logger.info('commencing model fitting')
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(inputs.nbatches):
                if MPI.COMM_WORLD.rank == 0:
                    logger.debug('batch = %d' % (batch))
                inputs_batch = ds.get_batch(inputs, batch)
                targets_batch = ds.get_batch(targets, batch)
                self.fprop(inputs_batch)
                self.bprop(targets_batch, inputs_batch, epoch)
                if MPI.COMM_WORLD.rank == 0:
                    error += self.cost.apply_function(self.backend,
                                                      self.layers[-1].output,
                                                      targets_batch,
                                                      self.temp)
            if MPI.COMM_WORLD.rank == 0:
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / inputs.nbatches))
            for layer in self.layers:
                logger.debug("%s", layer)

    def fprop(self, inputs):
        # call MLP's fprop: doesn't work for FC->FC connections
        # super(ConvnetDist, self).fprop(inputs)
        # handle FC-> FC connections
        y = inputs
        for layer in self.layers:
            if (isinstance(layer, LayerWithNoBiasDist) and
                isinstance(self.layers[layer.pos - 1],
                           LayerWithNoBiasDist)):
                y = y.take(layer.in_indices, axis=0)
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs, epoch):
        i = self.nlayers - 1
        lastlayer = self.layers[i]

        error = self.backend.zeros((self.layers[-1].nout, self.batch_size))
        # apply derivative on root node's FC layer output
        if MPI.COMM_WORLD.rank == 0:
            error = self.cost.apply_derivative(self.backend,
                                               lastlayer.output, targets,
                                               self.temp)
            self.backend.divide(error, self.backend.wrap(targets.shape[1]),
                                out=error)
        error._tensor = MPI.COMM_WORLD.bcast(error.raw())
        # Update the output layer.
        lastlayer.pre_act_ = lastlayer.pre_act
        while isinstance(self.layers[i], LayerWithNoBiasDist):
            if isinstance(self.layers[i - 1], LayerWithNoBiasDist):
                self.layers[i].bprop(error,
                                     self.layers[i - 1].output.
                                     take(self.layers[i].in_indices, axis=0),
                                     epoch)
            else:
                self.layers[i].bprop(error,
                                     self.layers[i - 1].output,
                                     epoch)
            error = self.layers[i].berror
            i -= 1
            if isinstance(self.layers[i], LayerWithNoBiasDist):
                # extract self.layers[i].pre_act terms
                self.layers[i].pre_act_ = self.layers[i].pre_act.take(
                    self.layers[i + 1].in_indices, axis=0)

        # following code is difficult to refactor:
        # 1) MPL berror has no halos for top layer, but does for middle layers
        # note: that input into MPL is ignored (self.layers[i -
        # 1].output)
        # Following is for top MPL layer
        self.layers[i].bprop(error,
                             self.layers[i - 1].output,
                             epoch)
        while i > 0:
            i -= 1
            # aggregate the berror terms at halo locations
            # note the MaxPoolingLayerDist ignores the input param
            # ConvLayerDist needs halo handling for input and berror
            self.layers[i].bprop(
                self.layers[
                    i + 1].input.local_array.get_bprop_view(
                    self.layers[i + 1].berror),
                self.layers[i].input.local_array.chunk,
                epoch)
