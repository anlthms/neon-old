"""
Generic distributed neural network layer built to handle data from a particular
backend.
"""

import logging

from neon.util.distarray.local_array import LocalArray
from neon.util.persist import YAMLable

logger = logging.getLogger(__name__)


class LocalLayerDist(YAMLable):

    """
    Base class for locally connected layers.
    """

    def adjust_for_halos(self, ifmshape):
        """
        ifmshape, rofmlocs etc. need to be updated
        after halos have been defined
        """
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape

        # cache the global array sizes
        self.global_ofmheight = self.ofmheight
        self.global_ofmwidth = self.ofmwidth
        self.global_ofmsize = self.ofmsize

        # local array sizes
        self.ofmheight = (self.ifmheight - self.fheight) / self.stride + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / self.stride + 1
        self.ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nin = self.nifm * self.ifmsize

        if self.pos > 0:
            self.berror = self.backend.zeros((self.batch_size, self.nin),
                                             dtype=self.dtype)

        ofmstarts = self.backend.array(range(0, (self.ofmsize * self.nofm),
                                             self.ofmsize))

        ofmlocs = self.backend.zeros((self.ofmsize, self.nofm), dtype='i32')
        for dst in xrange(self.ofmsize):
            ofmlocs[dst, :] = ofmstarts + dst
        # stores the flattened px location across
        # ofm in columns
        self.rofmlocs = ofmlocs.raw()

        # Figure out the connections with the previous layer.
        self.links = self.backend.zeros(
            (self.ofmsize, self.fsize), dtype='i32')
        # This variable tracks the top left corner of the receptive field.
        src = 0
        for dst in xrange(self.ofmsize):
            # Collect the column indices for the
            # entire receptive field.
            colinds = []
            for row in xrange(self.fheight):
                start = src + row * self.ifmwidth
                colinds += range(start, start + self.fwidth)
            fminds = colinds[:]
            for ifm in xrange(1, self.nifm):
                colinds += [x + ifm * self.ifmsize for x in fminds]

            if (src % self.ifmwidth + self.fwidth + self.stride) <= (
                    self.ifmwidth):
                # Slide the filter to the right by the stride value.
                src += self.stride
            else:
                # We hit the right edge of the input image.
                # Shift the filter down by one stride.
                src += self.stride * self.ifmwidth - src % self.ifmwidth
                assert src % self.ifmwidth == 0
            self.links[dst, :] = self.backend.array(colinds)
        self.rlinks = self.links.raw()

    def __init__(self, name, backend, batch_size, pos, learning_rate, nifm,
                 nofm, ifmshape, fshape, stride, dtype='float32'):
        self.name = name
        self.backend = backend
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape
        self.fheight, self.fwidth = fshape
        self.batch_size = batch_size
        self.pos = pos
        self.learning_rate = learning_rate
        self.pos = pos
        self.dtype = dtype
        self.ofmheight = (self.ifmheight - self.fheight) / stride + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / stride + 1
        self.ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        # self.nin = nifm * self.ifmsize
        # if pos > 0:
        #    self.berror = backend.zeros((batch_size, self.nin), dtype=dtype)

        self.nifm = nifm
        self.nofm = nofm
        self.fsize = nifm * self.fheight * self.fwidth
        self.stride = stride

    def normalize_weights(self, weights):
        norms = weights.norm(axis=1)
        self.backend.divide(weights,
                            norms.reshape((norms.shape[0], 1)),
                            out=weights)

    def fprop(self, inputs):
        raise NotImplementedError('This class should not be instantiated.')


class LocalFilteringLayerDist(LocalLayerDist):

    """
    Local filtering layer. This is very similar to ConvLayer, but the weights
    are not shared.
    """

    def adjust_for_halos(self, ifmshape, top_left_row_output,
                         top_left_col_output, dtype='float32'):
        super(LocalFilteringLayerDist, self).adjust_for_halos(ifmshape)
        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nout = self.ofmsize * self.nofm
        self.output = self.backend.zeros(
            (self.batch_size, self.nout), dtype=dtype)

        # if initializing the weights from scratch
        # self.weights = self.backend.gen_weights((self.nout, self.fsize),
        #                                        self.weight_init, dtype=dtype)

        # if initializing using same seed as non-dist version
        # adjust size of self.weights for halo dimensions
        out_indices = []
        for cur_channel in range(self.nofm):
            current_index = cur_channel * self.global_ofmsize + \
                top_left_row_output * self.global_ofmwidth + \
                top_left_col_output
            for cur_row in range(self.ofmheight):
                out_indices.extend(
                    range(current_index, current_index + self.ofmwidth))
                current_index += self.global_ofmwidth
        self.weights = self.weights.take(out_indices, axis=0)

        self.normalize_weights(self.weights)
        self.updates = self.backend.zeros(self.weights.shape, dtype=dtype)
        self.prodbuf = self.backend.zeros(
            (self.batch_size, self.nofm), dtype=dtype)
        self.bpropbuf = self.backend.zeros(
            (self.batch_size, self.fsize), dtype=dtype)
        self.updatebuf = self.backend.zeros(
            (self.nofm, self.fsize), dtype=dtype)

    def __init__(self, name, backend, batch_size, pos, learning_rate,
                 nifm, nofm, ifmshape, fshape, stride, weight_init,
                 pretraining, pretrain_learning_rate, sparsity, tied_weights):
        super(
            LocalFilteringLayerDist, self).__init__(name, backend, batch_size,
                                                    pos, learning_rate,
                                                    nifm, nofm, ifmshape,
                                                    fshape, stride)
        self.nout = self.ofmsize * nofm
        self.weight_init = weight_init
        self.weights = self.backend.gen_weights((self.nout, self.fsize),
                                                self.weight_init,
                                                dtype='float32')
        if pretraining is True:
            self.sparsity = sparsity
            self.pretrain_learning_rate = pretrain_learning_rate
            self.train_learning_rate = self.learning_rate
            self.tied_weights = tied_weights

    def __str__(self):
        return ("LocalFilteringLayer %s: %d ifms, "
                "utilizing %s backend\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nifm,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights)))

    def pretrain_mode(self, pooling):
        self.learning_rate = self.pretrain_learning_rate
        self.pooling = pooling
        self.defilter = LocalDeFilteringLayerDist(self, self.tied_weights)

    def train_mode(self):
        self.learning_rate = self.train_learning_rate

    def pretrain(self, inputs_dist, layer_id, cost, epoch, momentum):
        # Forward propagate the input through this layer and a
        # defiltering layer to reconstruct the input.
        inputs = inputs_dist[layer_id].local_array.chunk
        self.fprop(inputs)

        # todo: next 4 lines can be pre-initialized
        # for defiltering layer
        autoencoder = LocalArray(
            batch_size=self.batch_size,
            global_row_index=inputs_dist[
                layer_id].local_array.global_row_index,
            global_col_index=inputs_dist[
                layer_id].local_array.global_col_index,
            height=inputs_dist[layer_id].local_array.height,
            width=inputs_dist[layer_id].local_array.width,
            act_channels=inputs_dist[layer_id].local_array.act_channels,
            top_left_row=inputs_dist[layer_id].local_array.top_left_row,
            top_left_col=inputs_dist[layer_id].local_array.top_left_col,
            border_id=inputs_dist[layer_id].local_array.border_id,
            halo_size_row=inputs_dist[layer_id].local_array.halo_size_row,
            halo_size_col=inputs_dist[layer_id].local_array.halo_size_col,
            comm_per_dim=inputs_dist[layer_id].local_array.comm_per_dim,
            backend=self.backend)
        # reuse halo info from filtering layer
        autoencoder.send_halos = inputs_dist[layer_id].local_array.send_halos
        autoencoder.recv_halos = inputs_dist[layer_id].local_array.recv_halos
        autoencoder.local_image_indices = (
            inputs_dist[layer_id].local_array.local_image_indices)

        self.defilter.fprop(self.output)

        # halo aggregation across chunks for defiltering layer
        # todo: depending on how bprop uses defiltering_chunk
        # it may not need to belong to autoencoder
        # accumulate the reconstructed local_image patches
        autoencoder.defiltering_chunk = self.defilter.output
        autoencoder.send_recv_defiltering_layer_halos()
        autoencoder.make_defiltering_layer_consistent()

        # communicate the halos for the reconstructed image patches for bprop
        autoencoder.local_image = autoencoder.defiltering_local_image
        # autoencoder.chunk = autoencoder.defiltering_chunk #initialize
        autoencoder.send_recv_halos()
        autoencoder.make_local_chunk_consistent()

        # Forward propagate the output of this layer through a
        # pooling layer. The output of the pooling layer is used
        # to optimize sparsity.
        # MPI: set mini-batch to local_image
        inputs_dist[layer_id + 1].local_array.local_image = self.output
        # perform halo exchanges
        inputs_dist[layer_id + 1].local_array.send_recv_halos()
        # make consistent chunk
        inputs_dist[layer_id + 1].local_array.make_local_chunk_consistent()
        self.pooling.fprop(inputs_dist[layer_id + 1].local_array.chunk)

        # Backward propagate the gradient of the reconstruction error
        # through the defiltering layer.
        error = cost.apply_derivative(self.backend,
                                      autoencoder.chunk,
                                      inputs,
                                      self.defilter.temp)
        self.backend.divide(error, self.backend.wrap(inputs.shape[0]),
                            out=error)

        self.defilter.bprop(error, self.output, epoch, momentum)

        # Now backward propagate the gradient of the output of the
        # pooling layer.
        error = ((self.sparsity / inputs.shape[0]) *
                 (self.backend.ones(self.pooling.output.shape)))
        self.pooling.bprop(
            error, inputs_dist[layer_id + 1].local_array.chunk,
            epoch, momentum)

        # halo exchanges for the L2 pooling layer
        inputs_dist[
            layer_id + 1].local_array.defiltering_chunk = self.pooling.berror
        inputs_dist[
            layer_id + 1].local_array.send_recv_defiltering_layer_halos()
        inputs_dist[
            layer_id + 1].local_array.make_defiltering_layer_consistent()

        # Aggregate the errors from both layers before back propagating
        # through the current layer.
        berror = self.defilter.berror + (
            inputs_dist[layer_id + 1].local_array.defiltering_local_image)
        self.bprop(berror, inputs, epoch, momentum)

        rcost = cost.apply_function(self.backend,
                                    autoencoder.defiltering_local_image,
                                    inputs_dist[
                                        layer_id].local_array.local_image,
                                    self.defilter.temp1)
        spcost = self.sparsity * self.pooling.output.sum()
        return rcost, spcost

    def fprop(self, inputs):
        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            # We use a different filter for each receptive field.

            # size-guide
            # inputs.take: mbs x (ifmsize*nifm) ->  mbs x (fmsize*nifm)
            # self.weights: (nout x (ifmsize*nifm)).T -> (fsize x nofm)
            self.backend.dot(inputs.take(rflinks, axis=1),
                             self.weights.take(self.rofmlocs[dst], axis=0).T(),
                             out=self.prodbuf)

            # size: # mbs x nofm
            self.output[:, self.rofmlocs[dst]] = self.prodbuf

    def bprop(self, error, inputs, epoch, momentum):
        self.delta = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in xrange(self.ofmsize):
                # Use the same filter that was used for forward propagation
                # of this receptive field.

                # size-guide
                # self.delta.take: # mbs x nofm
                # self.weights.take: # (nofm x fsize )
                self.backend.dot(self.delta.take(self.rofmlocs[dst], axis=1),
                                 self.weights.take(
                                     self.rofmlocs[dst], axis=0),
                                 self.bpropbuf)
                # todo: the delta activations in halo terms have to be summed
                # when stacking the layers (for bprop)

                rflinks = self.rlinks[dst]
                self.backend.add(self.bpropbuf,  # mbs x fsize
                                 self.berror.take(
                                     rflinks, axis=1),  # mbs x fsize
                                 out=self.bpropbuf)
                self.berror[:, rflinks] = self.bpropbuf

        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            delta_slice = self.delta.take(
                self.rofmlocs[dst], axis=1)  # mbs x nofm
            self.backend.dot(delta_slice.T(),  # nofm x mbs
                             inputs.take(rflinks, axis=1),
                             # mbs x nifm
                             out=self.updatebuf)
            self.updates[
                self.rofmlocs[dst]] = self.updatebuf  # nofm x nifm

        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)
        self.normalize_weights(self.weights)


class LocalDeFilteringLayerDist(object):

    """
    Local defiltering layer. This reverses the actions
    of a local filtering layer.
    """

    def __init__(self, prev, tied_weights, dtype='float32'):
        self.output = prev.backend.zeros(
            (prev.batch_size, prev.nin), dtype=dtype)
        if tied_weights is True:
            # Share the weights with the previous layer.
            self.weights = prev.weights
        else:
            self.weights = prev.weights.copy()
        self.output = prev.backend.zeros(
            (prev.batch_size, prev.nin), dtype=dtype)
        self.updates = prev.backend.zeros(self.weights.shape, dtype=dtype)
        self.prodbuf = prev.backend.zeros(
            (prev.batch_size, prev.fsize), dtype=dtype)
        self.bpropbuf = prev.backend.zeros(
            (prev.batch_size, prev.nofm), dtype=dtype)
        self.updatebuf = prev.backend.zeros(
            (prev.nofm, prev.fsize), dtype=dtype)
        self.berror = prev.backend.zeros(
            (prev.batch_size, prev.nout), dtype=dtype)
        self.temp = [prev.backend.zeros(self.output.shape, dtype=dtype)]
        self.learning_rate = prev.pretrain_learning_rate
        self.backend = prev.backend
        self.rlinks = prev.rlinks
        self.prev = prev

    def fprop(self, inputs):
        self.backend.clear(self.output)
        for dst in xrange(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            # size guide:
            # inputs[:, self.prev.rofmlocs[dst]]: mbs x nout -> mbs x nofm
            # self.weights.take: nofm x ifmsize

            self.backend.dot(inputs[:, self.prev.rofmlocs[dst]],
                             self.weights.take(self.prev.rofmlocs[dst],
                                               axis=0),
                             out=self.prodbuf)
            self.output[:, rflinks] += self.prodbuf  # mbs x ifmsize

    def bprop(self, error, inputs, epoch, momentum):
        for dst in xrange(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(error[:, rflinks],  # mbs x ifmsize
                             self.weights.take(self.prev.rofmlocs[dst],
                                               axis=0).T(),  # ifmsize x nofm
                             out=self.bpropbuf)
            self.berror[:, self.prev.rofmlocs[dst]] = self.bpropbuf
            delta_slice = error[:, rflinks]
            self.backend.dot(inputs[:, self.prev.rofmlocs[dst]].T(),
                             delta_slice,
                             out=self.updatebuf)
            self.updates[self.prev.rofmlocs[dst]] = self.updatebuf
        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)
        self.prev.normalize_weights(self.weights)

# todo: add the L2 pooling layer code? for now in layer.py file under
# adjust_for_dist
