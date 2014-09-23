"""
Generic distributed neural network layer built to handle data from a particular
backend.
"""

import logging

import mylearn.util.distarray.localActArray as laa
from mylearn.util.persist import YAMLable

logger = logging.getLogger(__name__)


class LocalLayer_dist(YAMLable):

    """
    Base class for locally connected layers.
    """

    def adjustForHalos(self, ifmshape):
        """
        ifmshape, rofmlocs etc. need to be updated
        after halos have been defined
        """
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape

        self.ofmheight = (self.ifmheight - self.fheight) / self.stride + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / self.stride + 1
        self.ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nin = self.nifm * self.ifmsize

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
                 nofm, ifmshape, fshape, stride):
        self.name = name
        self.backend = backend
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape
        self.fheight, self.fwidth = fshape
        self.batch_size = batch_size
        self.pos = pos
        self.learning_rate = learning_rate

        # self.ofmheight = (self.ifmheight - self.fheight) / stride + 1
        # self.ofmwidth = (self.ifmwidth - self.fwidth) / stride + 1
        # self.ofmshape = (self.ofmheight, self.ofmwidth)
        # self.ifmsize = self.ifmheight * self.ifmwidth
        # self.ofmsize = self.ofmheight * self.ofmwidth
        # self.nin = nifm * self.ifmsize
        if pos > 0:
            self.berror = backend.zeros((batch_size, self.nin))

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


class LocalFilteringLayer_dist(LocalLayer_dist):

    """
    Local filtering layer. This is very similar to ConvLayer, but the weights
    are not shared.
    """

    def adjustForHalos(self, ifmshape):
        super(LocalFilteringLayer_dist, self).adjustForHalos(ifmshape)
        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nout = self.ofmsize * self.nofm
        self.output = self.backend.zeros((self.batch_size, self.nout))
        self.weights = self.backend.gen_weights((self.nout, self.fsize),
                                                self.weight_init)
        self.normalize_weights(self.weights)
        self.updates = self.backend.zeros(self.weights.shape)
        self.prodbuf = self.backend.zeros((self.batch_size, self.nofm))
        self.bpropbuf = self.backend.zeros((self.batch_size, self.fsize))
        self.updatebuf = self.backend.zeros((self.nofm, self.fsize))

    def __init__(self, name, backend, batch_size, pos, learning_rate,
                 nifm, nofm, ifmshape, fshape, stride, weight_init,
                 pretraining, pretrain_learning_rate, sparsity, tied_weights):
        super(
            LocalFilteringLayer_dist, self).__init__(name, backend, batch_size,
                                                     pos, learning_rate,
                                                     nifm, nofm, ifmshape,
                                                     fshape, stride)
        self.weight_init = weight_init
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
        self.defilter = LocalDeFilteringLayer_dist(self, self.tied_weights)

    def train_mode(self):
        self.learning_rate = self.train_learning_rate

    def pretrain(self, inputs_dist, cost, epoch, momentum):
        # Forward propagate the input through this layer and a
        # defiltering layer to reconstruct the input.
        inputs = inputs_dist.localActArray.chunk
        self.fprop(inputs)

        # todo: next 3 lines can be pre-initialized
        # for defiltering layer
        Y = laa.LocalActArray(
            batchSize=self.batch_size,
            globalRowIndex=inputs_dist.localActArray.globalRowIndex,
            globalColIndex=inputs_dist.localActArray.globalColIndex,
            height=self.ofmheight, width=self.ofmwidth, actChannels=self.nofm,
            topLeftRow=inputs_dist.localActArray.topLeftRow,
            topLeftCol=inputs_dist.localActArray.topLeftCol,
            borderId=inputs_dist.localActArray.borderId,
            haloSizeRow=-1, haloSizeCol=-1,
            commPerDim=inputs_dist.localActArray.commPerDim,
            backend=self.backend)
        # reuse halo info from filtering layer
        Y.sendHalos = inputs_dist.localActArray.sendHalos
        Y.recvHalos = inputs_dist.localActArray.recvHalos

        self.defilter.fprop(self.output)
        Y.localImage = self.output  # unused?
        # halo aggregation across chunks for defiltering layer

        # todo: depending on how bprop uses localImageDefiltering
        # it may not need to belong to Y
        Y.localImageDefiltering = self.defilter.output
        Y.sendRecvDefilteringLayerHalos()
        Y.makeDefilteringLayerConsistent()

        # use Y.localImageDefiltering for bprop
        self.defilter.output = Y.localImageDefiltering

        # Forward propagate the output of this layer through a
        # pooling layer. The output of the pooling layer is used
        # to optimize sparsity.
        self.pooling.fprop(self.output)

        # Backward propagate the gradient of the reconstruction error
        # through the defiltering layer.
        error = cost.apply_derivative(self.backend, self.defilter.output,
                                      inputs, self.defilter.temp)
        self.backend.divide(error, self.backend.wrap(inputs.shape[0]),
                            out=error)
        self.defilter.bprop(error, self.output, epoch, momentum)
        # Now backward propagate the gradient of the output of the
        # pooling layer.
        error = ((self.sparsity / inputs.shape[0]) *
                 (self.backend.ones(self.pooling.output.shape)))
        self.pooling.bprop(error, self.output, epoch, momentum)

        # Aggregate the errors from both layers before back propagating
        # through the current layer.
        berror = self.defilter.berror + self.pooling.berror

        self.bprop(berror, inputs, epoch, momentum)

        rcost = cost.apply_function(self.backend, self.defilter.output,
                                    inputs, self.defilter.temp)
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
                # when stacking the layers

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


class LocalDeFilteringLayer_dist(object):

    """
    Local defiltering layer. This reverses the actions
    of a local filtering layer.
    """

    def __init__(self, prev, tied_weights):
        self.output = prev.backend.zeros((prev.batch_size, prev.nin))
        if tied_weights is True:
            # Share the weights with the previous layer.
            self.weights = prev.weights
        else:
            self.weights = prev.weights.copy()
        self.updates = prev.backend.zeros(self.weights.shape)
        self.prodbuf = prev.backend.zeros((prev.batch_size, prev.fsize))
        self.bpropbuf = prev.backend.zeros((prev.batch_size, prev.nofm))
        self.updatebuf = prev.backend.zeros((prev.nofm, prev.fsize))
        self.berror = prev.backend.zeros((prev.batch_size, prev.nout))
        self.temp = [prev.backend.zeros(self.output.shape)]
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