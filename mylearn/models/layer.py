"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
from mylearn.transforms.gaussian import gaussian_filter
from mylearn.transforms.linear import Identity


logger = logging.getLogger(__name__)


class Layer(object):

    """
    Single NNet layer built to handle data from a particular backend
    """
    def __init__(self, name, backend, batch_size, pos, learning_rate, nin,
                 nout, activation, weight_init):
        self.name = name
        self.backend = backend
        self.weights = self.backend.gen_weights((nout, nin), weight_init)
        self.activation = activation
        self.nin = nin
        self.nout = nout
        self.velocity = self.backend.zeros(self.weights.shape)
        self.delta = backend.zeros((batch_size, nout))
        self.updates = backend.zeros((nout, nin))
        self.pre_act = backend.zeros((batch_size, self.nout))
        self.output = backend.zeros((batch_size, self.nout))
        self.pos = pos
        self.learning_rate = learning_rate
        if pos > 0:
            # This is storage for the backward propagated error.
            self.berror = backend.zeros((batch_size, nin - 1))

    def __str__(self):
        return ("Layer %s: %d inputs, %d nodes, %s act_fn, "
                "utilizing %s backend\n\t"
                "y: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "z: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t"
                "velocity: mean=%.05f, min=%.05f, max=%.05f" %
                (self.name, self.nin, self.nout,
                 self.activation.__class__.__name__,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.output),
                 self.backend.min(self.output),
                 self.backend.max(self.output),
                 self.backend.mean(self.pre_act),
                 self.backend.min(self.pre_act),
                 self.backend.max(self.pre_act),
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights),
                 self.backend.mean(self.velocity),
                 self.backend.min(self.velocity),
                 self.backend.max(self.velocity)))

    def fprop(self, inputs):
        inputs = self.backend.append_bias(inputs)
        self.backend.dot(inputs, self.weights.T(), out=self.pre_act)
        self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs, epoch, momentum):
        self.backend.multiply(error, self.pre_act, out=self.delta)
        if self.pos > 0:
            endcol = self.weights.shape[1] - 1
            self.backend.dot(self.delta, self.weights[:, 0:endcol],
                             out=self.berror)

        inputs = self.backend.append_bias(inputs)
        momentum_coef = self.backend.get_momentum_coef(epoch, momentum)
        self.backend.multiply(self.velocity, self.backend.wrap(momentum_coef),
                              out=self.velocity)
        self.backend.dot(self.delta.T(), inputs, out=self.updates)

        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.velocity, self.updates, out=self.velocity)
        self.backend.add(self.weights, self.velocity, out=self.weights)


class LayerWithNoBias(Layer):

    """
    Single NNet layer with no bias node - temporary code for testing purposes.
    """
    def __init__(self, name, backend, batch_size, pos, learning_rate, nin,
                 nout, activation, weight_init):
        super(LayerWithNoBias, self).__init__(name, backend, batch_size,
                                              pos, learning_rate, nin, nout,
                                              activation, weight_init)
        if pos > 0:
            self.berror = backend.zeros((batch_size, nin))

    def fprop(self, inputs):
        self.backend.dot(inputs, self.weights.T(), out=self.pre_act)
        if not isinstance(self.activation, Identity):
            self.activation.apply_both(self.backend, self.pre_act,
                                       self.output)

    def bprop(self, error, inputs, epoch, momentum):
        if not isinstance(self.activation, Identity):
            self.backend.multiply(error, self.pre_act, out=self.delta)
        else:
            self.delta = error
        if self.pos > 0:
            self.backend.dot(self.delta, self.weights, out=self.berror)

        self.backend.dot(self.delta.T(), inputs, out=self.updates)
        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)


class LayerWithNoActivation(LayerWithNoBias):
    def fprop(self, inputs):
        self.backend.dot(inputs, self.weights.T(), out=self.pre_act)

    def bprop(self, error, inputs, epoch, momentum):
        self.delta = error
        if self.pos > 0:
            self.backend.dot(self.delta, self.weights, out=self.berror)

        self.backend.dot(self.delta.T(), inputs, out=self.updates)
        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)


class RBMLayer(Layer):

    """
    CD1 training layer for RBM
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate, nin,
                 nout, activation, weight_init):
        super(RBMLayer, self).__init__(name, backend, batch_size, pos,
                                       learning_rate, nin, nout,
                                       activation, weight_init)
        self.p_hid_plus = backend.zeros((batch_size, self.nout))
        self.s_hid_plus = backend.zeros((batch_size, self.nout))
        self.p_hid_minus = backend.zeros((batch_size, self.nout))
        self.p_plus = backend.zeros((self.nout, nin))
        self.p_minus = backend.zeros((self.nout, nin))
        self.diff = backend.zeros((self.nout, nin))
        self.neg_pre_act = backend.zeros((batch_size, self.nin))
        self.x_minus = backend.zeros((batch_size, self.nin))

    def positive(self, inputs):
        """
        Positive / upward pass of the CD1 RBM

        Arguments:
           inputs (mylearn.datasets.dataset.Dataset): dataset upon which
                                                      to operate
        """
        inputs = self.backend.append_bias(inputs)
        self.backend.dot(inputs, self.weights.T(), out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_plus)
        self.backend.dot(self.p_hid_plus.T(), inputs, out=self.p_plus)
        self.random_numbers = self.backend.uniform(size=self.p_hid_plus.shape)
        self.backend.greater(self.p_hid_plus, self.random_numbers,
                             out=self.s_hid_plus)

    def negative(self, inputs):
        """
        Negative / downward pass of the CD1 RBM

        Arguments:
           inputs (mylearn.datasets.dataset.Dataset): dataset upon which
                                                      to operate
        """
        self.backend.dot(self.s_hid_plus, self.weights, out=self.neg_pre_act)
        self.activation.apply_function(self.backend, self.neg_pre_act,
                                       self.x_minus)
        self.backend.dot(self.x_minus, self.weights.T(), out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_minus)
        self.backend.dot(self.p_hid_minus.T(), self.x_minus,
                         out=self.p_minus)

    def update(self, epsilon, epoch, momentum):
        """
        CD1 weight update

        Arguments:
            epsilon: step size
            epoch: not used, for future compatibility
            momentum: not used, for future compatibility
        """
        self.backend.subtract(self.p_plus, self.p_minus, out=self.diff)
        self.backend.multiply(self.diff, self.backend.wrap(epsilon),
                              out=self.diff)
        self.backend.add(self.weights, self.diff, out=self.weights)
        # epoch, momentum?


class AELayer(LayerWithNoBias):
    """
    Single NNet layer built to handle data from a particular backend used
    in an Autoencoder.
    TODO: merge with generic Layer above.
    """
    def __init__(self, name, backend, batch_size, pos, learning_rate, nin,
                 nout, activation, weight_init, weights=None):
        super(AELayer, self).__init__(name, backend, batch_size, pos,
                                      learning_rate, nin, nout,
                                      activation, weight_init)
        if weights is not None:
            self.weights = weights


class LocalLayer(object):

    """
    Base class for locally connected layers.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate, nifm,
                 ifmshape, fshape, stride):
        self.name = name
        self.backend = backend
        self.ifmheight, self.ifmwidth = ifmshape
        self.fheight, self.fwidth = fshape
        self.batch_size = batch_size
        self.pos = pos
        self.learning_rate = learning_rate

        self.ofmheight = (self.ifmheight - self.fheight) / stride + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / stride + 1
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nin = nifm * self.ifmsize
        if pos > 0:
            self.berror = backend.zeros((batch_size, self.nin))

        self.nifm = nifm
        self.fsize = nifm * self.fheight * self.fwidth
        # Figure out the connections with the previous layer.
        self.links = backend.zeros((self.ofmsize, self.fsize), dtype='i32')
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
            for ifm in xrange(1, nifm):
                colinds += [x + ifm * self.ifmsize for x in fminds]

            if (src % self.ifmwidth + self.fwidth + stride) <= self.ifmwidth:
                # Slide the filter to the right by the stride value.
                src += stride
            else:
                # We hit the right edge of the input image.
                # Shift the filter down by one stride.
                src += stride * self.ifmwidth - src % self.ifmwidth
                assert src % self.ifmwidth == 0
            self.links[dst, :] = backend.array(colinds)
        self.rlinks = self.links.raw()

    def fprop(self, inputs):
        raise NotImplementedError('This class should not be instantiated.')


class ConvLayer(LocalLayer):

    """
    Convolutional layer.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate, nifm,
                 ifmshape, fshape, nfilt, stride, weight_init):
        super(ConvLayer, self).__init__(name, backend, batch_size, pos,
                                        learning_rate, nifm,
                                        ifmshape, fshape, stride)
        self.nout = self.ofmsize * nfilt
        self.nfilt = nfilt
        self.weights = backend.gen_weights((nfilt, self.fsize),
                                           weight_init)
        ofmstarts = backend.array(range(0, (self.ofmsize * nfilt),
                                        self.ofmsize))
        ofmlocs = backend.zeros((self.ofmsize, nfilt), dtype='i32')
        for dst in xrange(self.ofmsize):
            ofmlocs[dst, :] = ofmstarts + dst
        self.rofmlocs = ofmlocs.raw()

        self.output = backend.zeros((batch_size, self.nout))
        self.updates = backend.zeros(self.weights.shape)
        self.prodbuf = backend.zeros((batch_size, nfilt))
        self.bpropbuf = backend.zeros((batch_size, self.fsize))
        self.updatebuf = backend.zeros((nfilt, self.fsize))

    def __str__(self):
        return ("ConvLayer %s: %d ifms, %d filters, "
                "utilizing %s backend\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nifm, self.nfilt,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights)))

    def fprop(self, inputs):
        for dst in xrange(self.ofmsize):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            rflinks = self.rlinks[dst]
            self.backend.dot(inputs.take(rflinks, axis=1),
                             self.weights.T(), out=self.prodbuf)
            self.output[:, self.rofmlocs[dst]] = self.prodbuf

    def bprop(self, error, inputs, epoch, momentum):
        self.delta = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in xrange(self.ofmsize):
                self.backend.dot(self.delta.take(self.rofmlocs[dst], axis=1),
                                 self.weights, self.bpropbuf)
                rflinks = self.rlinks[dst]
                self.backend.add(self.bpropbuf,
                                 self.berror.take(rflinks, axis=1),
                                 out=self.bpropbuf)
                self.berror[:, rflinks] = self.bpropbuf

        self.backend.clear(self.updates)
        for dst in xrange(self.ofmsize):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = self.rlinks[dst]
            delta_slice = self.delta.take(self.rofmlocs[dst], axis=1)

            self.backend.dot(delta_slice.T(), inputs.take(rflinks, axis=1),
                             out=self.updatebuf)
            self.updates.add(self.updatebuf)
        # Update the filters after summing the weight updates.
        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)


class LocalFilteringLayer(LocalLayer):

    """
    Local filtering layer. This is very similar to ConvLayer, but the weights
    are not shared.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate,
                 nifm, ifmshape, fshape, stride, weight_init):
        super(LocalFilteringLayer, self).__init__(name, backend, batch_size,
                                                  pos, learning_rate,
                                                  nifm, ifmshape, fshape,
                                                  stride)
        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nout = self.ofmsize
        self.output = backend.zeros((batch_size, self.nout))
        self.weights = self.backend.gen_weights((self.ofmsize, self.fsize),
                                                weight_init)

        self.output = backend.zeros((batch_size, self.nout))
        self.updates = backend.zeros(self.weights.shape)
        self.prodbuf = backend.zeros((batch_size, 1))
        self.bpropbuf = backend.zeros((batch_size, self.fsize))
        self.recon = backend.zeros((batch_size, nifm * self.ifmsize))

    def __str__(self):
        return ("LocalFilteringLayer %s: %d ifms, "
                "utilizing %s backend\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nifm,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights)))

    def pretrain(self, inputs):
        # TODO
        pass

    def fprop(self, inputs):
        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            # We use a different filter for each receptive field.
            self.backend.dot(inputs.take(rflinks, axis=1),
                             self.weights[dst:(dst + 1)].T(),
                             out=self.prodbuf)
            self.output[:, dst:(dst + 1)] = self.prodbuf

    def bprop(self, error, inputs, epoch, momentum):
        self.delta = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in xrange(self.ofmsize):
                # Use the same filter that was used for forward propagation
                # of this receptive field.
                self.backend.dot(self.delta[:, dst:(dst + 1)],
                                 self.weights[dst:(dst + 1)],
                                 out=self.bpropbuf)
                rflinks = self.rlinks[dst]
                self.backend.add(self.bpropbuf,
                                 self.berror.take(rflinks, axis=1),
                                 out=self.bpropbuf)
                self.berror[:, rflinks] = self.bpropbuf

        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            delta_slice = self.delta[:, dst]
            self.backend.dot(delta_slice.T(),
                             inputs.take(rflinks, axis=1),
                             out=self.updates[dst])
        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)


class PoolingLayer(object):

    """
    Base class for pooling layers.
    """
    def __init__(self, name, backend, batch_size, pos,
                 nfm, ifmshape, pshape, stride):
        self.name = name
        self.backend = backend
        self.nfm = nfm
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.pheight, self.pwidth = pshape
        self.psize = self.pheight * self.pwidth
        self.pos = pos
        self.batch_size = batch_size

        ofmheight = (self.ifmheight - self.pheight) / stride + 1
        ofmwidth = (self.ifmwidth - self.pwidth) / stride + 1
        self.ofmsize = ofmheight * ofmwidth
        self.nin = nfm * self.ifmsize
        if pos > 0:
            self.berror = backend.zeros((batch_size, self.nin))

        # Figure out the possible connections with the previous layer.
        # Each unit in this layer could be connected to any one of
        # self.psize units in the previous layer.
        self.links = backend.zeros((self.ofmsize, self.psize), dtype='i32')
        # This variable tracks the top left corner of the receptive field.
        src = 0
        for dst in xrange(self.ofmsize):
            colinds = []
            # Collect the column indices for the
            # entire receptive field.
            for row in xrange(self.pheight):
                start = src + row * self.ifmwidth
                colinds += range(start, start + self.pwidth)
            if (src % self.ifmwidth + self.pwidth + stride) <= self.ifmwidth:
                # Slide the filter by the stride value.
                src += stride
            else:
                # We hit the right edge of the input image.
                # Shift the pooling window down by one stride.
                src += stride * self.ifmwidth - src % self.ifmwidth
                assert src % self.ifmwidth == 0
            self.links[dst, :] = backend.array(colinds)

        self.nout = nfm * self.ofmsize
        self.output = backend.zeros((batch_size, self.nout))
        self.delta = backend.zeros((batch_size, self.nout))

        # Reshape the matrices to have a single row per feature map.
        self.rdelta = self.backend.squish(self.delta, self.nfm)
        self.routput = self.backend.squish(self.output, self.nfm)
        if pos > 0:
            self.rberror = self.backend.squish(self.berror, self.nfm)

    def fprop(self, inputs):
        raise NotImplementedError('This class should not be instantiated.')


class MaxPoolingLayer(PoolingLayer):

    """
    Max pooling layer.
    """
    def __init__(self, name, backend, batch_size, pos, nfm, ifmshape, pshape,
                 stride):
        super(MaxPoolingLayer, self).__init__(name, backend, batch_size, pos,
                                              nfm, ifmshape, pshape, stride)
        self.maxinds = backend.zeros((batch_size * nfm, self.ofmsize),
                                     dtype='i32')

    def __str__(self):
        return ("MaxPoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t"
                "maxinds: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.maxinds),
                 self.backend.min(self.maxinds),
                 self.backend.max(self.maxinds)))

    def fprop(self, inputs):
        # Reshape the input so that we have a separate row
        # for each input feature map (this is to avoid a loop over
        # each feature map).
        inputs = self.backend.squish(inputs, self.nfm)
        for dst in xrange(self.ofmsize):
            # For this output unit, get the corresponding receptive fields
            # within all input feature maps.
            rf = inputs.take(self.links[dst], axis=1)
            # Save the index of the maximum value within the receptive fields.
            self.maxinds[:, dst] = rf.argmax(axis=1)
            # Set the pre-activations to the maximum value.
            maxvals = rf[range(rf.shape[0]), self.maxinds[:, dst]]
            self.routput[:, dst] = maxvals

    def bprop(self, error, inputs, epoch, momentum):
        self.delta[:] = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in xrange(self.ofmsize):
                links = self.links[dst]
                colinds = self.maxinds[:, dst]
                inds = links.take(colinds, axis=0)
                self.rberror[range(self.rberror.shape[0]), inds] += (
                    self.rdelta[:, dst])


class L2PoolingLayer(PoolingLayer):

    """
    L2 pooling layer. Each receptive field is pooled to obtain its L2 norm
    as output.
    """
    def __init__(self, name, backend, batch_size, pos, nfm, ifmshape, pshape,
                 stride):
        super(L2PoolingLayer, self).__init__(name, backend, batch_size, pos,
                                             nfm, ifmshape, pshape, stride)
        self.normalized_rf = backend.zeros((batch_size * nfm, self.ifmsize))
        self.prodbuf = backend.zeros((batch_size * nfm, self.psize))

    def __str__(self):
        return ("L2PoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        rinputs = self.backend.squish(inputs, self.nfm)
        for dst in xrange(self.ofmsize):
            inds = self.links[dst]
            rf = rinputs.take(inds, axis=1)
            self.routput[:, dst] = rf.norm(axis=1)
            denom = self.routput[:, dst:(dst + 1)].repeat(self.psize, axis=1)
            # If the L2 norm is zero, the entire receptive field must be zeros.
            # In that case, we set the L2 norm to 1 before using it to
            # normalize the receptive field.
            denom[denom == 0] = 1
            self.normalized_rf[:, inds] = rf / denom

    def bprop(self, error, inputs, epoch, momentum):
        self.delta[:] = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in xrange(self.ofmsize):
                inds = self.links[dst]
                self.backend.multiply(self.normalized_rf[:, inds],
                                      self.rdelta[:, dst:(dst + 1)],
                                      out=self.prodbuf)
                self.rberror[:, inds] += self.prodbuf


class AveragePoolingLayer(PoolingLayer):

    """
    Average pooling.
    """

    def __init__(self, name, backend, batch_size, pos, nfm, ifmshape, pshape,
                 stride):
        super(AveragePoolingLayer, self).__init__(name, backend, batch_size,
                                                  pos, nfm, ifmshape, pshape,
                                                  stride)
        self.nout = nfm * self.ofmsize

    def __str__(self):
        return ("AveragePoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        rinputs = self.backend.squish(inputs, self.nfm)
        for dst in range(self.ofmsize):
            inds = self.links[dst]
            rf = rinputs.take(inds, axis=1)
            self.routput[:, dst] = rf.mean(axis=1)

    def bprop(self, error, inputs, epoch, momentum):
        self.delta[:] = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            self.rdelta /= self.psize
            for dst in range(self.ofmsize):
                inds = self.links[dst]
                self.rberror[:, inds] += (self.rdelta.take(range(dst, dst + 1),
                                          axis=1))


class LCNLayer(LocalLayer):

    """
    Local contrast normalization.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(LCNLayer, self).__init__(name, backend, batch_size, pos, 0.0,
                                       nifm, ifmshape, fshape, stride)
        self.nin = nifm * self.ifmsize
        self.nout = nifm * self.ifmsize
        self.output = backend.zeros((batch_size, self.nin))
        self.filter = self.normalized_gaussian_filters(nifm, fshape)
        self.meanfm = self.backend.zeros((self.batch_size,
                                          nifm * self.ofmsize))
        self.ex_meanfm = self.backend.zeros((self.batch_size, self.ifmheight,
                                             self.ifmwidth))
        self.rex_meanfm = self.ex_meanfm.reshape((self.batch_size, self.nin))
        self.inset_row = self.ifmheight - self.ofmheight
        self.inset_col = self.ifmwidth - self.ofmwidth

        self.prodbuf = backend.zeros((batch_size, 1))
        self.output = backend.zeros((batch_size, self.nout))
        self.intermed = backend.zeros(self.output.shape)
        self.delta = backend.zeros((batch_size, self.nout))
        self.rdelta = self.backend.squish(self.delta, self.nifm)
        self.denom = self.backend.zeros((self.batch_size, self.nin))
        if pos > 0:
            self.rberror = self.backend.squish(self.berror, self.nifm)

    def __str__(self):
        return ("LCNLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def normalized_gaussian_filters(self, count, shape):
        """
        Return multiple copies of gaussian filters with values adding up to
        one.
        """

        filter = gaussian_filter(shape)
        filter /= (count * filter.sum())
        return self.backend.wrap(filter.reshape((filter.shape[0] *
                                                 filter.shape[1], 1)))

    def sub_normalize(self, inputs):
        # Convolve with gaussian filters to obtain a "mean" feature map.
        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(inputs.take(rflinks, axis=1), self.filter,
                             out=self.prodbuf)
            self.meanfm[:, dst:(dst + 1)] = self.prodbuf

        # TODO: handle edges better.
        self.rmeanfm = self.meanfm.reshape((self.batch_size, self.ofmheight,
                                            self.ofmwidth))
        for row in xrange(self.ex_meanfm.shape[0]):
            self.ex_meanfm[row,
                           self.inset_row:(self.inset_row + self.ofmheight),
                           self.inset_col:(self.inset_col + self.ofmwidth)
                           ] = (self.rmeanfm[row])

        self.intermed[:] = inputs
        for i in xrange(self.nifm):
            self.intermed[:, i * self.ifmsize:(i + 1) * self.ifmsize] -= (
                self.rex_meanfm)

    def div_normalize(self):
        self.backend.multiply(self.intermed, self.intermed, out=self.output)
        self.backend.clear(self.denom)
        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(self.output.take(rflinks, axis=1), self.filter,
                             out=self.prodbuf)
            self.denom[:, dst:(dst + 1)] = self.prodbuf
        self.backend.sqrt(self.denom, out=self.denom)
        c = self.denom.mean()
        self.denom[self.denom < c] = c
        self.backend.divide(self.intermed, self.denom, out=self.output)

    def fprop(self, inputs):
        self.sub_normalize(inputs)
        self.div_normalize()

    def bprop(self, error, inputs, epoch, momentum):
        self.delta[:] = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            self.backend.divide(self.rdelta, self.backend.wrap(self.fsize),
                                self.rdelta)
            for dst in xrange(self.ofmsize):
                links = self.links[dst]
                self.rberror[:, links] += self.rdelta[:, dst:(dst + 1)]
