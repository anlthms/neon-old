"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging

from neon.transforms.gaussian import gaussian_filter
from neon.util.persist import YAMLable
import neon.util.distarray.gdist_consts as gc

logger = logging.getLogger(__name__)


class Layer(YAMLable):

    """
    Single NNet layer built to handle data from a particular backend

    Attributes:
        name (str): Used to identify this layer when logging.
        backend (neon.backends.backend.Backend): underlying type for stored
                                                    data parameters like
                                                    weights.
        batch_size (int): Number of examples presented at each iteration
        pos (int): The layers position (0-based)
        weights (neon.backends.backend.Tensor): weight values associated
                                                   with each node.
        activation (neon.transforms.activation.Activation): activation
                   function to apply to each node during a forward propogation
        nin (int): Number of inputs to this layer.
        nout (int): Number of outputs from this layer.
        output (neon.backends.backend.Tensor): final transformed output
                                                  values from this layer.
        pre_act (neon.backends.backend.Tensor): intermediate node values
                                                   from this layer prior
                                                   to applying activation
                                                   transform.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate, nin,
                 nout, activation, weight_init, weight_dtype=None,
                 velocity_dtype=None, delta_dtype=None, updates_dtype=None,
                 pre_act_dtype=None, output_dtype=None, berror_dtype=None):
        self.name = name
        self.backend = backend
        self.activation = activation
        self.nin = nin
        self.nout = nout
        self.weight_init = weight_init
        self.weight_dtype = weight_dtype
        self.velocity_dtype = velocity_dtype
        self.weights = self.backend.gen_weights((nout, nin), weight_init,
                                                weight_dtype)

        self.velocity = self.backend.zeros(self.weights.shape, velocity_dtype)
        self.delta = self.backend.alloc(batch_size, nout, delta_dtype)
        self.updates = self.backend.zeros((nout, nin), updates_dtype)
        self.updates_dtype = updates_dtype
        self.pre_act = self.backend.alloc(batch_size, self.nout,
                                          pre_act_dtype)
        self.output = self.backend.alloc(batch_size, self.nout, output_dtype)
        self.pos = pos
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        if pos > 0:
            # This is storage for the backward propagated error.
            self.berror = self.backend.alloc(batch_size, nin - 1,
                                             berror_dtype)
            self.berror_dtype = berror_dtype

    def __str__(self):
        return ("Layer {lyr_nm}: {nin} inputs, {nout} nodes, {act_nm} act_fn, "
                "utilizing {be_nm} backend\n\t"
                "y: mean={y_avg:g}, min={y_min:g}, abs_min={y_absmin:g}, "
                "max={y_max:g},\n\t"
                "   dtype={y_dtype}\n\t"
                "z: mean={z_avg:g}, min={z_min:g}, abs_min={z_absmin:g}, "
                "max={z_max:g},\n\t"
                "   dtype={z_dtype}\n\t"
                "weights: mean={w_avg:g}, min={w_min:g}, abs_min={w_absmin:g},"
                " max={w_max:g},\n\t"
                "         dtype={w_dtype}\n\t"
                "velocity: mean={v_avg:g}, min={v_min:g}, "
                "abs_min={v_absmin:g}, max={w_max:g},\n\t"
                "          dtype={v_dtype}\n".format
                (lyr_nm=self.name, nin=self.nin, nout=self.nout,
                 act_nm=self.activation.__class__.__name__,
                 be_nm=self.backend.__class__.__name__,
                 y_avg=self.backend.mean(self.output),
                 y_min=self.backend.min(self.output),
                 y_absmin=self.backend.min(self.backend.fabs(self.output)),
                 y_max=self.backend.max(self.output),
                 y_dtype=self.output.dtype,
                 z_avg=self.backend.mean(self.pre_act),
                 z_min=self.backend.min(self.pre_act),
                 z_absmin=self.backend.min(self.backend.fabs(self.pre_act)),
                 z_max=self.backend.max(self.pre_act),
                 z_dtype=self.pre_act.dtype,
                 w_avg=self.backend.mean(self.weights),
                 w_min=self.backend.min(self.weights),
                 w_absmin=self.backend.min(self.backend.fabs(self.weights)),
                 w_max=self.backend.max(self.weights),
                 w_dtype=self.weights.dtype,
                 v_avg=self.backend.mean(self.velocity),
                 v_min=self.backend.min(self.velocity),
                 v_absmin=self.backend.min(self.backend.fabs(self.velocity)),
                 v_max=self.backend.max(self.velocity),
                 v_dtype=self.velocity.dtype))

    def fprop(self, inputs):
        inputs = self.backend.append_bias(inputs)
        self.backend.fprop_fc_dot(inputs, self.weights, out=self.pre_act)
        self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs, epoch, momentum):
        self.backend.multiply(error, self.pre_act, out=self.delta)
        if self.pos > 0:
            endcol = self.weights.shape[1] - 1
            self.backend.bprop_fc_dot(self.delta, self.weights[:, 0:endcol],
                                      out=self.berror)

        inputs = self.backend.append_bias(inputs)
        momentum_coef = self.backend.get_momentum_coef(epoch, momentum)
        self.backend.multiply(self.velocity, self.backend.wrap(momentum_coef),
                              out=self.velocity)
        self.backend.update_fc_dot(self.delta, inputs, out=self.updates)

        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.velocity, self.updates, out=self.velocity)
        self.backend.add(self.weights, self.velocity, out=self.weights)


class LayerWithNoBias(Layer):

    """
    Single NNet layer with no bias node
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate, nin,
                 nout, activation, weight_init, weight_dtype=None,
                 velocity_dtype=None, delta_dtype=None, updates_dtype=None,
                 pre_act_dtype=None, output_dtype=None, berror_dtype=None):
        super(LayerWithNoBias, self).__init__(name, backend, batch_size,
                                              pos, learning_rate, nin, nout,
                                              activation, weight_init,
                                              weight_dtype, velocity_dtype,
                                              delta_dtype, updates_dtype,
                                              pre_act_dtype, output_dtype,
                                              berror_dtype)
        if pos > 0:
            self.berror = backend.alloc(batch_size, nin)

    def fprop(self, inputs):
        self.backend.fprop_fc_dot(inputs, self.weights, out=self.pre_act)
        self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs, epoch, momentum):
        # comment if not using denominator term in cross_entropy
        self.backend.multiply(error, self.pre_act, out=self.delta)
        # self.delta = error

        if self.pos > 0:
            self.backend.bprop_fc_dot(self.delta, self.weights,
                                      out=self.berror)

        self.backend.update_fc_dot(self.delta, inputs, out=self.updates)
        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)


class LayerWithNoBiasDist(LayerWithNoBias):

    """
    MPI Distributed
    Single NNet layer with no bias node
    """

    def adjust_for_dist(self, nin, ifmshape, nifm, global_size, global_width,
                        top_left_row_output, top_left_col_output):
        self.nin = nin
        out_indices = []
        # ifmsize = ifmshape[0] * ifmshape[1]
        for cur_channel in range(nifm):
            current_index = cur_channel * global_size + \
                top_left_row_output * global_width + top_left_col_output
            for cur_row in range(ifmshape[0]):
                out_indices.extend(
                    range(current_index, current_index + ifmshape[1]))
                current_index += global_width
        self.weights = self.weights.take(out_indices, axis=1)

        self.velocity = self.backend.zeros(
            self.weights.shape, self.velocity_dtype)
        self.updates = self.backend.zeros((self.nout, nin), self.updates_dtype)
        if self.pos > 0:
            # This is storage for the backward propagated error.
            self.berror = self.backend.zeros((self.batch_size, nin),
                                             self.berror_dtype)

    def fprop(self, inputs):
        self.backend.dot(inputs, self.weights.T(), out=self.pre_act)

    def fprop2(self):
        # this stores the derivatives in self.pre_act
        self.activation.apply_both(self.backend, self.pre_act, self.output)


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
           inputs (neon.datasets.dataset.Dataset): dataset upon which
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
           inputs (neon.datasets.dataset.Dataset): dataset upon which
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


class LocalLayer(YAMLable):

    """
    Base class for locally connected layers.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate, nifm,
                 nofm, ifmshape, fshape, stride):
        self.name = name
        self.backend = backend
        self.batch_size = batch_size
        self.pos = pos
        self.learning_rate = learning_rate
        self.nifm = nifm
        self.nofm = nofm
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape
        self.fheight, self.fwidth = fshape
        self.stride = stride

        self.ofmheight = (self.ifmheight - self.fheight) / stride + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / stride + 1
        self.ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nin = nifm * self.ifmsize
        if pos > 0:
            self.berror = backend.zeros((batch_size, self.nin))

        self.fsize = nifm * self.fheight * self.fwidth
        ofmstarts = backend.array(range(0, (self.ofmsize * nofm),
                                        self.ofmsize)).raw()
        ofmlocs = backend.zeros((self.ofmsize, nofm), dtype='i32')
        for dst in xrange(self.ofmsize):
            ofmlocs[dst, :] = backend.wrap(ofmstarts + dst)
        self.rofmlocs = ofmlocs.raw()

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
            self.links[dst, :] = backend.array(colinds, dtype='i32')
        self.rlinks = self.links.raw()

    def normalize_weights(self, weights):
        norms = weights.norm(axis=1)
        self.backend.divide(weights,
                            norms.reshape((norms.shape[0], 1)),
                            out=weights)

    def fprop(self, inputs):
        raise NotImplementedError('This class should not be instantiated.')


class ConvLayer(LocalLayer):

    """
    Convolutional layer.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate, nifm,
                 nofm, ifmshape, fshape, stride, weight_init):
        super(ConvLayer, self).__init__(name, backend, batch_size, pos,
                                        learning_rate, nifm, nofm,
                                        ifmshape, fshape, stride)
        self.nout = self.ofmsize * nofm
        self.weights = backend.gen_weights((nofm, self.fsize),
                                           weight_init)
        self.output = backend.zeros((batch_size, self.nout))
        self.updates = backend.zeros(self.weights.shape)
        self.prodbuf = backend.zeros((batch_size, nofm))
        self.bpropbuf = backend.zeros((batch_size, self.fsize))
        self.updatebuf = backend.zeros((nofm, self.fsize))

    def __str__(self):
        return ("ConvLayer %s: %d ifms, %d filters, "
                "utilizing %s backend\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nifm, self.nofm,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights)))

    def fprop(self, inputs):
        self.backend.fprop_conv(self.weights, inputs, self.output,
                                self.rlinks, self.ifmshape, self.ofmshape,
                                self.rofmlocs, 0, self.stride, self.nifm, 1,
                                self.prodbuf)

    def bprop(self, error, inputs, epoch, momentum):
        if self.pos > 0:
            self.backend.bprop_conv(self.weights, error, self.berror,
                                    self.links, self.ofmshape, self.rofmlocs,
                                    self.bpropbuf)
        self.backend.update_conv(self.weights, inputs, error, self.updates,
                                 self.links, self.ofmshape, self.rofmlocs,
                                 self.learning_rate, self.updatebuf)


class LocalFilteringLayer(LocalLayer):

    """
    Local filtering layer. This is very similar to ConvLayer, but the weights
    are not shared.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate,
                 nifm, nofm, ifmshape, fshape, stride, weight_init,
                 pretraining, pretrain_learning_rate, sparsity, tied_weights):
        super(LocalFilteringLayer, self).__init__(name, backend, batch_size,
                                                  pos, learning_rate,
                                                  nifm, nofm, ifmshape, fshape,
                                                  stride)
        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nout = self.ofmsize * nofm
        self.output = backend.zeros((batch_size, self.nout))
        self.weights = self.backend.gen_weights((self.nout, self.fsize),
                                                weight_init)

        self.normalize_weights(self.weights)
        self.updates = backend.zeros(self.weights.shape)
        self.prodbuf = backend.zeros((batch_size, nofm))
        self.bpropbuf = backend.zeros((batch_size, self.fsize))
        self.updatebuf = backend.zeros((nofm, self.fsize))
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
        self.defilter = LocalDeFilteringLayer(self, self.tied_weights)

    def train_mode(self):
        self.learning_rate = self.train_learning_rate

    def pretrain(self, inputs, cost, epoch, momentum):
        # Forward propagate the input through this layer and a
        # defiltering layer to reconstruct the input.
        self.fprop(inputs)
        self.defilter.fprop(self.output)
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
            self.backend.dot(inputs.take(rflinks, axis=1),
                             self.weights.take(self.rofmlocs[dst], axis=0).T(),
                             out=self.prodbuf)
            self.output[:, self.rofmlocs[dst]] = self.prodbuf

    def bprop(self, error, inputs, epoch, momentum):
        self.delta = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in xrange(self.ofmsize):
                # Use the same filter that was used for forward propagation
                # of this receptive field.
                self.backend.dot(self.delta.take(self.rofmlocs[dst], axis=1),
                                 self.weights.take(self.rofmlocs[dst], axis=0),
                                 self.bpropbuf)
                rflinks = self.rlinks[dst]
                self.backend.add(self.bpropbuf,
                                 self.berror.take(rflinks, axis=1),
                                 out=self.bpropbuf)
                self.berror[:, rflinks] = self.bpropbuf

        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            delta_slice = self.delta.take(self.rofmlocs[dst], axis=1)
            self.backend.dot(delta_slice.T(),
                             inputs.take(rflinks, axis=1),
                             out=self.updatebuf)
            self.updates[self.rofmlocs[dst]] = self.updatebuf

        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)
        self.normalize_weights(self.weights)


class LocalDeFilteringLayer(object):

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
            self.backend.dot(inputs[:, self.prev.rofmlocs[dst]],
                             self.weights.take(self.prev.rofmlocs[dst],
                                               axis=0),
                             out=self.prodbuf)
            self.output[:, rflinks] += self.prodbuf

    def bprop(self, error, inputs, epoch, momentum):
        for dst in xrange(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(error[:, rflinks],
                             self.weights.take(self.prev.rofmlocs[dst],
                                               axis=0).T(),
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


class PoolingLayer(YAMLable):

    """
    Base class for pooling layers.
    """

    def adjust_for_dist(self, ifmshape, dtype='float32'):
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmsize = self.ifmheight * self.ifmwidth

        ofmheight = (self.ifmheight - self.pheight) / self.stride + 1
        ofmwidth = (self.ifmwidth - self.pwidth) / self.stride + 1
        self.ofmsize = ofmheight * ofmwidth
        self.nin = self.nfm * self.ifmsize
        self.ofmshape = [ofmheight, ofmwidth]
        if self.pos > 0:
            self.berror = self.backend.zeros((self.batch_size, self.nin),
                                             dtype)

        # Figure out the possible connections with the previous layer.
        # Each unit in this layer could be connected to any one of
        # self.psize units in the previous layer.
        self.links = self.backend.zeros(
            (self.ofmsize, self.psize), dtype='i32')
        # This variable tracks the top left corner of the receptive field.
        src = 0
        for dst in xrange(self.ofmsize):
            colinds = []
            # Collect the column indices for the
            # entire receptive field.
            for row in xrange(self.pheight):
                start = src + row * self.ifmwidth
                colinds += range(start, start + self.pwidth)
            if (src % self.ifmwidth + self.pwidth + self.stride) <= (
                    self.ifmwidth):
                # Slide the filter by the stride value.
                src += self.stride
            else:
                # We hit the right edge of the input image.
                # Shift the pooling window down by one stride.
                src += self.stride * self.ifmwidth - src % self.ifmwidth
                assert src % self.ifmwidth == 0
            self.links[dst, :] = self.backend.array(colinds)

        self.nout = self.nfm * self.ofmsize
        self.output = self.backend.zeros((self.batch_size, self.nout), dtype)
        self.delta = self.backend.zeros((self.batch_size, self.nout), dtype)

        # setup reshaped view variables
        self._init_reshaped_views()

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
        self.stride = stride

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
            self.links[dst, :] = backend.array(colinds, dtype='i32')

        self.nout = nfm * self.ofmsize
        self.output = backend.zeros((batch_size, self.nout))
        self.delta = backend.zeros((batch_size, self.nout))

        # setup reshaped view variables
        self._init_reshaped_views()

    def _init_reshaped_views(self):
        """
        Initialize reshaped view references to the arrays such that there is a
        single row per feature map.
        """
        self.rdelta = self.backend.squish(self.delta, self.nfm)
        self.routput = self.backend.squish(self.output, self.nfm)
        if self.pos > 0:
            self.rberror = self.backend.squish(self.berror, self.nfm)

    def __getstate__(self):
        """
        Fine-grained control over the serialization to disk of an instance of
        this class.
        """
        # since we will load a shared memory view, prevent writing reshaped
        # object copies to disk
        res = self.__dict__.copy()
        res['rdelta'] = None
        res['routput'] = None
        if self.pos > 0:
            res['rberror'] = None
        return res

    def __setstate__(self, state):
        """
        Fine-grained control over the loading of serialized object
        representation of this class.

        In this case we need to ensure that reshaped view references are
        restored (copies of these variables are created when writing to disk)

        Arguments:
            state (dict): keyword attribute values to be loaded.
        """
        self.__dict__.update(state)
        self._init_reshaped_views()

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
                "maxinds: mean=%.05f, min=%.05f, max=%.05f\n\t"
                "output: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.maxinds),
                 self.backend.min(self.maxinds),
                 self.backend.max(self.maxinds),
                 self.backend.mean(self.output),
                 self.backend.min(self.output),
                 self.backend.max(self.output)))

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
        self.prodbuf = self.backend.zeros((batch_size * nfm, self.psize))

    def adjust_for_dist(self, ifmshape, dtype='float32'):
        super(L2PoolingLayer, self).adjust_for_dist(ifmshape)
        self.prodbuf = self.backend.zeros(
            (self.batch_size * self.nfm, self.psize), dtype)

    def __str__(self):
        return ("L2PoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        rinputs = self.backend.squish(inputs, self.nfm)
        # print MPI.COMM_WORLD.rank, 'L2 pooling fprop rinputs.shape',
        # rinputs.shape
        for dst in xrange(self.ofmsize):
            inds = self.links[dst]
            rf = rinputs.take(inds, axis=1)
            self.routput[:, dst] = rf.norm(axis=1)

    def bprop(self, error, inputs, epoch, momentum):
        self.delta[:] = error
        rinputs = self.backend.squish(inputs, self.nfm)
        # print MPI.COMM_WORLD.rank, 'L2 pooling bprop rinputs.shape',
        # rinputs.shape
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in xrange(self.ofmsize):
                inds = self.links[dst]
                rf = rinputs.take(inds, axis=1)
                denom = self.routput[:, dst:(dst + 1)].copy()
                # If the L2 norm is zero, the entire receptive field must be
                # zeros. In that case, we set the L2 norm to 1 before using
                # it to normalize the receptive field.
                denom[denom.raw() == 0] = 1
                self.backend.divide(rf, denom, out=rf)
                self.backend.multiply(
                    self.rdelta[:, dst:(dst + 1)].repeat(self.psize, axis=1),
                    rf, out=self.prodbuf)
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


class Convolver(LocalLayer):

    """
    Lightweight convolutional layer that only does fprop.
    """

    def __init__(self, backend, batch_size, nifm,
                 nofm, ifmshape, fshape, stride, weights, dtype='float'):
        super(Convolver, self).__init__('conv', backend, batch_size, 0,
                                        0.0, nifm, nofm,
                                        ifmshape, fshape, stride)
        self.nout = self.ofmsize * nofm
        self.weights = weights
        self.output = backend.zeros((batch_size, self.nout), dtype)
        self.prodbuf = backend.zeros((batch_size, nofm), dtype)

    def fprop(self, inputs):
        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(inputs.take(rflinks, axis=1),
                             self.weights.T(), out=self.prodbuf)
            self.output[:, self.rofmlocs[dst]] = self.prodbuf


class LCNLayer(YAMLable):

    """
    Local contrast normalization.
    """

    def adjust_for_dist(self, ifmshape, border_id=-1,
                        output_height=-1, output_width=-1,
                        inputs_dist=None, dtype='float32'):
        self.dist_flag = True
        self.ifmshape = ifmshape
        self.ifmheight, self.ifmwidth = ifmshape  # with halos, but not padding
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.nin = self.nfm * self.ifmsize
        self.nout = output_height * output_width * self.nfm
        self.filters = self.normalized_gaussian_filters(
            self.nfm, self.fshape, dtype='float32')
        self.inputs_dist = inputs_dist

        if border_id != gc.CENTER:
            pad_height = self.fheight - 1
            pad_width = self.fwidth - 1

            # compute how much to pad
            pad_width_left = pad_width // 2
            pad_width_right = pad_width - pad_width_left
            pad_height_top = pad_height // 2
            pad_height_bottom = pad_height - pad_height_top

            left_padding = 0
            right_padding = 0
            top_padding = 0
            bottom_padding = 0
            self.start_row = 0  # top left corner after padded area (excl halo)
            self.start_col = 0

            if border_id in [gc.NORTH, gc.NORTHWEST, gc.NORTHEAST]:
                top_padding = pad_height_top
                self.start_row = top_padding
            if border_id in [gc.SOUTH, gc.SOUTHWEST, gc.SOUTHEAST]:
                bottom_padding = pad_height_bottom
            if border_id in [gc.WEST, gc.NORTHWEST, gc.SOUTHWEST]:
                left_padding = pad_width_left
                self.start_col = left_padding
            if border_id in [gc.EAST, gc.NORTHEAST, gc.SOUTHEAST]:
                right_padding = pad_width_right

        # todo: only supports stride of 1 for now
        self.exifmheight = (self.ifmheight) * self.stride + (
            top_padding + bottom_padding)
        self.exifmwidth = (self.ifmwidth) * self.stride + (
            left_padding + right_padding)
        self.exifmsize = self.exifmheight * self.exifmwidth
        self.exifmshape = (self.exifmheight, self.exifmwidth)

        self.exinputs = self.backend.zeros((self.batch_size,
                                            self.nfm * self.exifmsize), dtype)
        self.rexinputs = self.exinputs.reshape((self.batch_size, self.nfm,
                                                self.exifmheight,
                                                self.exifmwidth))
        self.conv = Convolver(self.backend, self.batch_size, self.nfm, 1,
                              self.exifmshape, self.fshape, self.stride,
                              self.filters, dtype)
        # assert self.conv.ofmsize == self.ifmsize

        self.hdiff = self.exifmheight - output_height
        self.wdiff = self.exifmwidth - output_width
        assert self.hdiff % 2 == 0
        assert self.wdiff % 2 == 0
        self.start_row2 = self.hdiff / 2  # top left corner for halo + padding
        self.start_col2 = self.wdiff / 2

        self.meanfm = self.conv.output
        self.rmeanfm = self.meanfm.reshape((self.batch_size, 1,
                                            output_height, output_width))

        self.output = self.backend.zeros((self.batch_size, self.nout), dtype)
        self.routput = self.output.reshape((self.batch_size, self.nfm,
                                            output_height, output_width))

        self.temp1 = self.backend.zeros(self.output.shape, dtype)
        self.rtemp1 = self.temp1.reshape(self.routput.shape)
        self.temp2 = self.backend.zeros(self.output.shape, dtype)
        self.rtemp2 = self.temp2.reshape(self.routput.shape)
        self.subout = self.backend.zeros(self.output.shape, dtype)
        self.rsubout = self.subout.reshape(self.routput.shape)
        self.subtemp = self.backend.zeros(self.output.shape)
        self.rsubtemp = self.subtemp.reshape(self.routput.shape)
        self.subtemp2 = self.backend.zeros((self.batch_size, self.nin), dtype)
        self.rsubtemp2 = self.subtemp2.reshape((self.batch_size, self.nfm,
                                                self.ifmheight, self.ifmwidth))

        if self.pos > 0:
            # changed to nout for bprop in dist version, compared to nin in
            # non-dist version
            self.diverror = self.backend.zeros(
                (self.batch_size, self.nout), dtype)
            self.exerror = self.backend.zeros((self.batch_size,
                                              self.nfm * self.exifmsize),
                                              dtype)
            self.rexerror = self.exerror.reshape((self.batch_size, self.nfm,
                                                  self.exifmheight,
                                                  self.exifmwidth))
            self.prodbuf = self.backend.zeros(
                (self.batch_size, self.fsize), dtype)
            self.bprop_filters = self.backend.zeros((self.nfm,
                                                    self.filters.shape[0],
                                                    self.filters.shape[1]),
                                                    dtype)
            self.sqtemp = self.backend.zeros(self.output.shape, dtype)
            for fm in xrange(self.nfm):
                self.bprop_filters[fm] = self.filters.copy()
                rfilter = self.bprop_filters[fm].reshape(
                    (self.nfm, self.fheight, self.fwidth))
                rfilter[fm, self.fheight / 2, self.fwidth / 2] -= 1.0

    def __init__(self, name, backend, batch_size, pos, nfm, ifmshape, fshape,
                 stride, dist_flag=False):
        self.name = name
        self.backend = backend
        self.ifmshape = ifmshape
        self.ifmheight, self.ifmwidth = ifmshape
        self.fheight, self.fwidth = fshape
        self.fsize = nfm * self.fheight * self.fwidth
        self.batch_size = batch_size
        self.nfm = nfm
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.nin = nfm * self.ifmsize
        self.nout = self.nin
        self.dist_flag = dist_flag

        self.filters = self.normalized_gaussian_filters(nfm, fshape)
        # self.fpeakdiff = 1.0 - self.fpeak
        self.stride = stride
        self.fshape = fshape
        self.pos = pos

        self.exifmheight = (self.ifmheight - 1) * stride + self.fheight
        self.exifmwidth = (self.ifmwidth - 1) * stride + self.fwidth
        self.exifmsize = self.exifmheight * self.exifmwidth
        self.exifmshape = (self.exifmheight, self.exifmwidth)

        self.exinputs = self.backend.zeros((batch_size, nfm * self.exifmsize))
        self.rexinputs = self.exinputs.reshape((self.batch_size, self.nfm,
                                                self.exifmheight,
                                                self.exifmwidth))
        self.conv = Convolver(backend, batch_size, nfm, 1,
                              self.exifmshape, fshape, stride,
                              self.filters)
        assert self.conv.ofmsize == self.ifmsize

        self.hdiff = self.exifmheight - self.ifmheight
        self.wdiff = self.exifmwidth - self.ifmwidth
        assert self.hdiff % 2 == 0
        assert self.wdiff % 2 == 0
        self.start_row = self.hdiff / 2
        self.start_col = self.wdiff / 2

        self.meanfm = self.conv.output
        self.rmeanfm = self.meanfm.reshape((batch_size, 1,
                                            self.ifmheight,
                                            self.ifmwidth))

        self.output = backend.zeros((batch_size, self.nout))
        self.routput = self.output.reshape((batch_size, nfm,
                                            self.ifmheight,
                                            self.ifmwidth))
        self.subout = backend.zeros(self.output.shape)
        self.rsubout = self.subout.reshape(self.routput.shape)
        self.subtemp = backend.zeros(self.output.shape)
        self.rsubtemp = self.subtemp.reshape(self.routput.shape)
        if pos > 0:
            self.diverror = backend.zeros((batch_size, self.nin))
            self.exerror = self.backend.zeros((batch_size,
                                               nfm * self.exifmsize))
            self.rexerror = self.exerror.reshape((batch_size, nfm,
                                                  self.exifmheight,
                                                  self.exifmwidth))
            self.prodbuf = self.backend.zeros((batch_size, self.fsize))
            self.bprop_filters = self.backend.zeros((nfm,
                                                    self.filters.shape[0],
                                                    self.filters.shape[1]))
            self.sqtemp = backend.zeros(self.output.shape)
            for fm in xrange(nfm):
                self.bprop_filters[fm] = self.filters.copy()
                rfilter = self.bprop_filters[fm].reshape(
                    (nfm, self.fheight, self.fwidth))
                rfilter[fm, self.fheight / 2, self.fwidth / 2] -= 1.0

    def __str__(self):
        return ("LCNLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def normalized_gaussian_filters(self, count, shape, dtype='float'):
        """
        Return multiple copies of gaussian filters with values adding up to
        one.
        """
        assert(len(shape) == 2)
        single = gaussian_filter(shape)
        single /= (count * single.sum())
        assert shape[0] % 2 == 1
        assert shape[1] % 2 == 1
        filters = self.backend.zeros((count, shape[0], shape[1]), dtype)
        filters[:] = single

        filters = filters.reshape((1, count * shape[0] * shape[1]))
        return filters

    def copy_to_inset(self, canvas, inset, start_row, start_col):
        if self.dist_flag:
            canvas[:, :,
                   start_row:start_row + inset.shape[2],
                   start_col:start_col + inset.shape[3]] = inset
        else:
            canvas[:, :,
                   start_row:(canvas.shape[2] - start_row),
                   start_col:(canvas.shape[3] - start_col)] = inset

    def copy_from_inset(self, canvas, start_row, start_col):
        if self.dist_flag:
            return canvas[:, :,
                          start_row:start_row + self.ifmheight,
                          start_col:start_col + self.ifmwidth]
        else:
            return canvas[:, :,
                          self.start_row:(canvas.shape[2] - start_row),
                          self.start_col:(canvas.shape[3] - start_col)]

    def fprop_sub_normalize(self, inputs):
        rinputs = inputs.reshape((self.batch_size, self.nfm,
                                  self.ifmheight, self.ifmwidth))
        self.copy_to_inset(self.rexinputs, rinputs,
                           self.start_row, self.start_col)

        # Convolve with gaussian filters to obtain a "mean" feature map.
        self.conv.fprop(self.exinputs)

        if self.dist_flag:
            # rinputs includes halos but not padding
            self.backend.subtract(
                self.rexinputs[:, :,
                               self.start_row2:(
                                   self.rexinputs.shape[2] - self.start_row2),
                               self.start_col2:(
                                   self.rexinputs.shape[3] - self.start_col2)],
                self.rmeanfm,
                out=self.rsubout)
        else:
            self.backend.subtract(rinputs, self.rmeanfm, out=self.rsubout)

    def fprop_div_normalize(self):
        if self.dist_flag:
            self.backend.multiply(self.inputs_dist.local_array.chunk,
                                  self.inputs_dist.local_array.chunk,
                                  out=self.subtemp2)
            self.copy_to_inset(self.rexinputs, self.rsubtemp2,
                               self.start_row, self.start_col)
        else:
            self.backend.multiply(self.subout, self.subout, out=self.subtemp)
            self.copy_to_inset(self.rexinputs, self.rsubtemp,
                               self.start_row, self.start_col)

        self.conv.fprop(self.exinputs)
        self.backend.sqrt(self.meanfm, out=self.meanfm)
        assert self.subout[self.meanfm.raw() == 0.0].sum() == 0.0
        self.meanfm[self.meanfm.raw() == 0.0] = 1.0
        if self.dist_flag:
            self.backend.divide(
                self.inputs_dist.local_array.local_image.reshape(
                    self.routput.shape), self.rmeanfm, out=self.routput)
        else:
            self.backend.divide(self.rsubout, self.rmeanfm, out=self.routput)

    def fprop(self, inputs):
        self.backend.clear(self.exinputs)

        self.fprop_sub_normalize(inputs)
        if self.dist_flag:
            # distributed version
            self.inputs_dist.local_array.local_image = self.subout
            self.inputs_dist.local_array.send_recv_halos()
            self.inputs_dist.local_array.make_local_chunk_consistent()

        self.fprop_div_normalize()

    def reshape_error(self):
        # discards zero padding around the delta matrix
        self.berror = self.copy_from_inset(self.rexerror, self.start_row,
                                           self.start_col)
        self.berror = self.berror.reshape((self.batch_size, self.nin))

    def bprop_sub_normalize(self, error, inputs, epoch, momentum):
        self.backend.clear(self.exerror)
        for fm in range(self.nfm):
            for dst in xrange(self.conv.ofmsize):
                rflinks = self.conv.rlinks[dst]
                loc = self.conv.rofmlocs[dst] + self.conv.ofmsize * fm
                filt = self.bprop_filters[fm]
                self.backend.multiply(error[:, loc], filt, out=self.prodbuf)
                self.exerror[:, rflinks] -= self.prodbuf
        self.reshape_error()

    def bprop_div_normalize(self, error, inputs, epoch, momentum):
        self.backend.clear(self.exerror)
        self.backend.cube(self.output, out=self.diverror)

        if self.dist_flag:
            self.subout = self.inputs_dist.local_array.local_image
            self.subtemp2[:] = self.inputs_dist.local_array.chunk

        self.subtemp[:] = self.subout
        assert self.diverror[self.subout.raw() == 0].sum() == 0.0
        self.subout[self.subout.raw() == 0] = 1.0
        self.backend.square(self.subout, out=self.sqtemp)
        # this is for the non-padded, non-halo matrix only
        self.backend.divide(self.diverror, self.sqtemp, out=self.diverror)

        for fm in range(self.nfm):
            for dst in xrange(self.conv.ofmsize):
                # self.conv.rofmlocs is over 1 fm only
                loc = self.conv.rofmlocs[dst] + self.conv.ofmsize * fm
                divout = self.output.take(loc, axis=1)
                subout = self.subout.take(loc, axis=1)
                assert divout[subout.raw() == 0].sum() == 0
                subout[subout.raw() == 0.0] = 1.0
                self.backend.divide(divout, subout, out=divout)

                rflinks = self.conv.rlinks[dst]
                if self.dist_flag:
                    self.copy_to_inset(self.rexinputs, self.rsubtemp2,
                                       self.start_row, self.start_col)
                else:
                    self.copy_to_inset(self.rexinputs, self.rsubtemp,
                                       self.start_row, self.start_col)
                rrexinputs = self.rexinputs.reshape(
                    (self.batch_size, self.nfm * self.exifmsize))
                frame = rrexinputs.take(rflinks, axis=1)
                self.backend.multiply(frame, self.filters, out=frame)
                self.backend.multiply(frame, self.diverror[:, loc], out=frame)
                rframe = frame.reshape((self.batch_size, self.nfm,
                                        self.fheight, self.fwidth))
                # this is working on the g2/y2 term
                rframe[:, fm:(fm + 1),
                       self.fheight / 2, self.fwidth / 2] -= divout
                self.backend.multiply(error[:, loc].repeat(self.fsize, axis=1),
                                      frame, out=frame)
                self.exerror[:, rflinks] -= frame
        self.reshape_error()

    def bprop(self, error, inputs, epoch, momentum):
        if self.pos > 0:
            # note: have to account for halos + padding after each step
            self.bprop_div_normalize(error, inputs, epoch, momentum)

            if self.dist_flag:
                self.inputs_dist.local_array.defiltering_chunk = self.berror
                self.inputs_dist.local_array.send_recv_defiltering_layer_halos(
                )
                self.inputs_dist.local_array.make_defiltering_layer_consistent(
                )
                self.bprop_sub_normalize(
                    self.inputs_dist.local_array.defiltering_local_image,
                    inputs, epoch, momentum)
                self.inputs_dist.local_array.defiltering_chunk = self.berror
                self.inputs_dist.local_array.send_recv_defiltering_layer_halos(
                )
                self.inputs_dist.local_array.make_defiltering_layer_consistent(
                )
                self.berror = (
                    self.inputs_dist.local_array.defiltering_local_image)
            else:
                self.bprop_sub_normalize(self.berror, inputs, epoch, momentum)

    def bprop_fast(self, error, inputs, epoch, momentum):
        """
        An incorrect, but much faster version of backprop.
        """
        if self.pos > 0:
            self.berror[:] = error
