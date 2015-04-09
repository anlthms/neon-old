# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Convolution network using halopar
"""

import logging

from neon.models.deprecated.mlp_dist import MLPDist
from neon.layers.deprecated.layer import ConvLayerDist, MaxPoolingLayerDist
from neon.layers.deprecated.layer import LayerDist
from neon.util.compat import range
from neon.util.distarray.global_array import GlobalArray

logger = logging.getLogger(__name__)


class ConvnetDist(MLPDist):

    """
    Halo/tower distributed convolutional network
    """

    def adjust_for_dist(self):
        # MPI: call adjust_for_dist for each layer
        layer = self.layers[0]
        layer.input = GlobalArray(cur_layer=layer)
        layer.adjust_for_dist()
        for i in range(1, self.nlayers):
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
            elif isinstance(layer, LayerDist):
                # fully connected layer: no halo transfers needed
                # nout_ is the full size of the layer
                # nout will be the split size of the layer
                layer.nout_ = layer.nout
                if i < self.nlayers - 1:
                    if layer.nout % self.backend.mpi_size != 0:
                        raise ValueError('Unsupported layer.nout % '
                                         'MPI.COMM_WORLD.size != 0')
                    layer.nout = layer.nout / self.backend.mpi_size
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
                    logger.debug('global_size=%d, global_width=%d',
                                 layer.global_size, layer.global_width)
                    layer.nin = mp_layer.nout
                    layer.ifmshape = mp_layer.ofmshape
                    layer.nifm = mp_layer.nifm
                    layer.prev_layer = 'MaxPoolingLayerDist'
                elif isinstance(self.layers[i - 1], LayerDist):
                    # split the inputs nin across MPI.COMM_WORLD.size
                    if layer.nin % self.backend.mpi_size != 0:
                        raise ValueError('Unsupported layer.nin % '
                                         'MPI.COMM_WORLD.size != 0')
                    layer.nin = layer.nin / self.backend.mpi_size
                    layer.in_indices = range(self.backend.mpi_rank * layer.nin,
                                             (self.backend.mpi_rank + 1) *
                                             layer.nin)
                    layer.prev_layer = 'LayerDist'
                else:
                    raise ValueError('Unsupported previous layer for '
                                     'LayerDist')
            layer.adjust_for_dist()

        if self.num_epochs > 0:
            # MPI related initializations for supervised bprop
            self.agg_output = self.backend.zeros(
                self.layers[-1].output.shape, 'float32')
            self.error = self.backend.zeros(
                (self.layers[-1].nout, self.batch_size))

    def fprop(self, inputs):
        # call MLP's fprop: doesn't work for FC->FC connections
        # super(ConvnetDist, self).fprop(inputs)
        # handle FC-> FC connections
        y = inputs
        for layer in self.layers:
            if (isinstance(layer, LayerDist) and
                isinstance(self.layers[layer.pos - 1],
                           LayerDist)):
                y = y.take(layer.in_indices, axis=0)
            layer.fprop(y)
            y = layer.output

    def bprop(self, targets, inputs):
        i = self.nlayers - 1
        lastlayer = self.layers[i]

        error = self.backend.zeros((self.layers[-1].nout, self.batch_size))
        # apply derivative on root node's FC layer output
        if self.backend.mpi_rank == 0:
            error = self.cost.apply_derivative(targets)
            self.backend.divide(error, targets.shape[1], out=error)
        error._tensor = self.backend.comm.bcast(error.asnumpyarray())
        # Update the output layer.
        lastlayer.pre_act_ = lastlayer.pre_act
        while isinstance(self.layers[i], LayerDist):
            if isinstance(self.layers[i - 1], LayerDist):
                self.layers[i].bprop(error,
                                     self.layers[i - 1].output.
                                     take(self.layers[i].in_indices, axis=0))
            else:
                self.layers[i].bprop(error,
                                     self.layers[i - 1].output)
            error = self.layers[i].deltas
            i -= 1
            if isinstance(self.layers[i], LayerDist):
                # extract self.layers[i].pre_act terms
                self.layers[i].pre_act_ = self.layers[i].pre_act.take(
                    self.layers[i + 1].in_indices, axis=0)

        # following code is difficult to refactor:
        # 1) MPL deltas has no halos for top layer, but does for middle layers
        # note: that input into MPL is ignored (self.layers[i -
        # 1].output)
        # Following is for top MPL layer
        self.layers[i].bprop(error, self.layers[i - 1].output)
        while i > 0:
            i -= 1
            # aggregate the deltas terms at halo locations
            # note the MaxPoolingLayerDist ignores the input param
            # ConvLayerDist needs halo handling for input and deltas
            self.layers[i].bprop(
                self.layers[
                    i + 1].input.local_array.get_bprop_view(
                    self.layers[i + 1].deltas),
                self.layers[i].input.local_array.chunk)