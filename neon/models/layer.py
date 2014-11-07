"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
import numpy as np

from neon.transforms.gaussian import gaussian_filter
from neon.util.compat import MPI_INSTALLED
from neon.util.distarray import gdist_consts as gc
from neon.util.distarray.local_array import LocalArray
from neon.util.persist import YAMLable

if MPI_INSTALLED:
    from mpi4py import MPI

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

    def __init__(self, name, backend, batch_size, pos, nin, nout,
                 activation, weight_init, learning_rule, weight_dtype=None,
                 delta_dtype=None, updates_dtype=None, pre_act_dtype=None,
                 output_dtype=None, berror_dtype=None):
        self.name = name
        self.backend = backend
        self.activation = activation
        self.nin = nin
        self.nout = nout
        self.weight_init = weight_init
        self.weight_dtype = weight_dtype
        self.weights = self.backend.gen_weights((nout, nin), weight_init,
                                                weight_dtype)
        self.delta = self.backend.alloc(batch_size, nout, delta_dtype)
        self.updates = self.backend.zeros((nout, nin), updates_dtype)
        self.updates_dtype = updates_dtype
        self.pre_act = self.backend.alloc(batch_size, self.nout,
                                          pre_act_dtype)
        self.output = self.backend.alloc(batch_size, self.nout, output_dtype)
        self.pos = pos
        self.learning_rule = learning_rule
        self.learning_rule.allocate_state(self.updates)
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
                "         dtype={w_dtype}\n".format
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
                 w_dtype=self.weights.dtype))

    def fprop(self, inputs):
        inputs = self.backend.append_bias(inputs)
        self.backend.fprop_fc_dot(inputs, self.weights, out=self.pre_act)
        self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs, epoch):
        """
        # numpy pseudocode for the backprop:
        # updates = dot(delta.transpose(), inputs)  # calculate new gradient
        # weight update itself done by application of learning rule
        """
        self.backend.multiply(error, self.pre_act, out=self.delta)
        if self.pos > 0:
            endcol = self.weights.shape[1] - 1
            self.backend.bprop_fc_dot(self.delta, self.weights[:, 0:endcol],
                                      out=self.berror)

        inputs = self.backend.append_bias(inputs)
        self.backend.update_fc_dot(self.delta, inputs, out=self.updates)
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class LayerWithNoBias(Layer):

    """
    Single NNet layer with no bias node
    """

    def __init__(self, name, backend, batch_size, pos, nin, nout,
                 activation, weight_init, learning_rule, weight_dtype=None,
                 delta_dtype=None, updates_dtype=None, pre_act_dtype=None,
                 output_dtype=None, berror_dtype=None):
        super(LayerWithNoBias, self).__init__(name, backend, batch_size,
                                              pos, nin, nout, activation,
                                              weight_init, learning_rule)
        if pos > 0:
            self.berror = backend.alloc(batch_size, nin)

    def fprop(self, inputs):
        self.backend.fprop_fc_dot(inputs, self.weights, out=self.pre_act)
        self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs, epoch):
        # comment if not using denominator term in cross_entropy
        self.backend.multiply(error, self.pre_act, out=self.delta)
        if self.pos > 0:
            self.backend.bprop_fc_dot(self.delta, self.weights,
                                      out=self.berror)
        self.backend.update_fc_dot(self.delta, inputs, out=self.updates)

        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class LayerWithNoBiasDist(LayerWithNoBias):

    """
    MPI Distributed
    Single NNet layer with no bias node
    """

    def adjust_for_dist(self):
        # indices of the input layer in weight matrix
        out_indices = []
        cond1 = self.prev_layer == 'MaxPoolingLayerDist'
        cond2 = self.prev_layer == 'LCNLayerDist'
        if cond1 or cond2:
            logger.debug('ifmshape[0]=%d, ifmshape[1]=%d, nifm=%d, '
                         'global_size=%d, global_width=%d', self.ifmshape[0],
                         self.ifmshape[1], self.nifm, self.global_size,
                         self.global_width)
            for cur_channel in range(self.nifm):
                current_index = (cur_channel * self.global_size +
                                 self.top_left_row_output * self.global_width +
                                 self.top_left_col_output)
                for cur_row in range(self.ifmshape[0]):
                    out_indices.extend(
                        range(current_index, current_index + self.ifmshape[1]))
                    current_index += self.global_width
        elif self.prev_layer == 'LayerWithNoBiasDist':
            out_indices = self.out_indices
        else:
            raise ValueError('Unsupported previous layer for '
                             'LayerWithNoBiasDist')

        self.weights = self.weights.take(out_indices, axis=1)

        self.updates = self.backend.zeros(self.weights.shape)
        self.learning_rule.allocate_state(self.updates)
        self.delta = self.backend.zeros((self.batch_size, self.nout))
        self.delta_ = self.backend.zeros((self.batch_size, self.nout_))
        self.delta_gather = self.backend.zeros(
            (self.batch_size * MPI.COMM_WORLD.size, self.nout))
        if self.pos > 0:
            # This is storage for the backward propagated error.
            self.berror = self.backend.zeros((self.batch_size, self.nin))

    def fprop(self, inputs):
        # dot product is distributed across nodes
        self.backend.fprop_fc_dot(inputs, self.weights, out=self.pre_act)
        # accumulate the pre_act values before applying non-linearity
        self.pre_act._tensor = MPI.COMM_WORLD.reduce(
            self.pre_act.raw(), op=MPI.SUM, root=0)
        # apply non-linearity on the output node
        if MPI.COMM_WORLD.rank == 0:
            # this stores the derivatives in self.pre_act
            self.activation.apply_both(self.backend, self.pre_act, self.output)
        # strictly, following line not needed for top-most layer
        self.output._tensor = MPI.COMM_WORLD.bcast(self.output.raw())
        # broadcast back the pre_act values for bprop.
        # note: suboptimal for dist implementation,
        # but a consequence of reusing the pre_act buffer for fprop and bprop
        self.pre_act._tensor = MPI.COMM_WORLD.bcast(self.pre_act.raw())

    def bprop(self, error, inputs, epoch):
        # comment if not using denominator term in cross_entropy
        self.backend.multiply(error, self.pre_act_, out=self.delta)
        if self.nout_ != self.nout:
            MPI.COMM_WORLD.Allgather(
                self.delta.raw(), self.delta_gather._tensor)
            # todo: only supported in numpy backend for now
            self.delta_._tensor = np.hstack(
                np.split(self.delta_gather.raw(), MPI.COMM_WORLD.size))
            if self.pos > 0:
                self.backend.bprop_fc_dot(self.delta_, self.weights,
                                          out=self.berror)
            self.backend.update_fc_dot(self.delta_, inputs, out=self.updates)
        else:
            if self.pos > 0:
                self.backend.bprop_fc_dot(self.delta, self.weights,
                                          out=self.berror)
            self.backend.update_fc_dot(self.delta, inputs, out=self.updates)

        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class RBMLayer(Layer):

    """
    CD1 training layer for RBM
    """

    def __init__(self, name, backend, batch_size, pos, nin,
                 nout, activation, weight_init, learning_rule):
        super(RBMLayer, self).__init__(name, backend, batch_size, pos,
                                       nin, nout, activation, weight_init,
                                       learning_rule)
        self.p_hid_plus = backend.alloc(batch_size, self.nout)
        self.s_hid_plus = backend.alloc(batch_size, self.nout)
        self.p_hid_minus = backend.alloc(batch_size, self.nout)
        self.p_plus = backend.zeros((self.nout, nin))
        self.p_minus = backend.zeros((self.nout, nin))
        self.diff = backend.zeros((self.nout, nin))
        self.learning_rule = learning_rule
        self.learning_rule.allocate_state(self.diff)
        self.neg_pre_act = backend.alloc(batch_size, self.nin)
        self.x_minus = backend.alloc(batch_size, self.nin)

    def positive(self, inputs):
        """
        Positive / upward pass of the CD1 RBM

        Arguments:
           inputs (neon.datasets.dataset.Dataset): dataset upon which
                                                      to operate
        """
        inputs = self.backend.append_bias(inputs)
        self.backend.fprop_fc_dot(inputs, self.weights, out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_plus)
        self.backend.update_fc_dot(self.p_hid_plus, inputs, out=self.p_plus)
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
        self.backend.bprop_fc_dot(self.s_hid_plus, self.weights,
                                  out=self.neg_pre_act)
        self.activation.apply_function(self.backend, self.neg_pre_act,
                                       self.x_minus)
        self.backend.fprop_fc_dot(self.x_minus, self.weights, out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_minus)
        self.backend.update_fc_dot(self.p_hid_minus, self.x_minus,
                                   out=self.p_minus)

    def update(self, epoch):
        """
        CD1 weight update

        Arguments:
            epoch: not used, for future compatibility
        """
        self.backend.subtract(self.p_plus, self.p_minus, out=self.diff)

        self.learning_rule.apply_rule(self.weights, self.diff, epoch)
        # epoch, momentum?


class AELayer(LayerWithNoBias):

    """
    Single NNet layer built to handle data from a particular backend used
    in an Autoencoder.
    TODO: merge with generic Layer above.
    """

    def __init__(self, name, backend, batch_size, pos, nin,
                 nout, activation, weight_init, learning_rule, weights=None):
        super(AELayer, self).__init__(name, backend, batch_size, pos,
                                      nin, nout, activation, weight_init,
                                      learning_rule)
        if weights is not None:
            self.weights = weights


class LocalLayer(YAMLable):

    """
    Base class for locally connected layers.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, pooling=False, activation=None):
        self.name = name
        self.backend = backend
        self.activation = activation
        self.batch_size = batch_size
        self.pos = pos
        self.nifm = nifm
        self.nofm = nofm
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape
        self.fshape = fshape
        self.fheight, self.fwidth = fshape
        self.stride = stride
        self.learning_rule = learning_rule

        self.ofmheight = (self.ifmheight - self.fheight) / stride + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / stride + 1
        self.ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nin = nifm * self.ifmsize
        if pos > 0:
            self.berror = backend.alloc(batch_size, self.nin)

        self.fsize = nifm * self.fheight * self.fwidth
        ofmstarts = backend.array(range(0, (self.ofmsize * nofm),
                                        self.ofmsize)).raw()
        self.ofmlocs = backend.zeros((self.ofmsize, nofm), dtype='i32')
        for dst in xrange(self.ofmsize):
            self.ofmlocs[dst, :] = backend.wrap(ofmstarts + dst)

        # Figure out the connections with the previous layer.
        if pooling is True:
            self.links = backend.zeros(
                (self.ofmsize, fshape[0] * fshape[1]), dtype='i32')
        else:
            self.links = backend.zeros(
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
            if pooling is False:
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


class LocalLayerDist(LocalLayer):

    """
    Base class for locally connected layers.
    """

    def adjust_for_dist(self, ifmshape):
        """
        ifmshape, ofmlocs etc. need to be updated
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
            self.berror = self.backend.zeros((self.batch_size, self.nin))

        ofmstarts = self.backend.array(range(0, (self.ofmsize * self.nofm),
                                             self.ofmsize))

        self.ofmlocs = self.backend.zeros((self.ofmsize, self.nofm),
                                          dtype='i32')
        for dst in xrange(self.ofmsize):
            self.ofmlocs[dst, :] = ofmstarts + dst
        # stores the flattened px location across
        # ofm in columns

        # Figure out the connections with the previous layer.
        if self.pooling is True:
            self.links = self.backend.zeros(
                (self.ofmsize, self.fshape[0] * self.fshape[1]), dtype='i32')
        else:
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
            if self.pooling is False:
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

        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.zeros((self.batch_size, self.nout))

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, pooling=False):
        self.name = name
        self.backend = backend
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape
        self.fshape = fshape
        self.fheight, self.fwidth = fshape
        self.batch_size = batch_size
        self.pos = pos
        self.learning_rule = learning_rule
        self.pos = pos
        # self.dtype = dtype
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
        self.pooling = pooling


class ConvLayer(LocalLayer):

    """
    Convolutional layer.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, weight_init, activation=None):
        super(ConvLayer, self).__init__(name, backend, batch_size, pos,
                                        learning_rule, nifm, nofm,
                                        ifmshape, fshape, stride,
                                        activation=activation)
        self.nout = self.ofmsize * nofm
        self.weights = backend.gen_weights((self.fsize, nofm),
                                           weight_init)
        self.output = backend.alloc(batch_size, self.nout)
        self.updates = backend.zeros(self.weights.shape)
        self.prodbuf = backend.alloc(batch_size, nofm)
        self.bpropbuf = backend.alloc(batch_size, self.fsize)
        self.updatebuf = backend.zeros(self.weights.shape)
        self.learning_rule.allocate_state(self.updates)
        if activation is not None:
            self.pre_act = backend.alloc(batch_size, self.nout)
        else:
            self.pre_act = self.output 

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
        self.backend.fprop_conv(self.weights, inputs, self.pre_act,
                                self.rlinks, self.ifmshape, self.ofmshape,
                                self.ofmlocs, 0, self.stride, self.nifm, 1,
                                self.prodbuf)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs, epoch):
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)
        if self.pos > 0:
            self.backend.bprop_conv(self.weights, error, self.berror,
                                    self.links, self.ifmshape, self.ofmshape,
                                    self.ofmlocs, 0, self.stride, self.nifm,
                                    1, self.bpropbuf)
        self.backend.update_conv(self.weights, inputs, error, self.updates,
                                 self.links, self.ifmshape, self.ofmshape,
                                 self.ofmlocs, 0, self.stride, self.nifm,
                                 1, self.fwidth, self.updatebuf)
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class ConvLayerDist(LocalLayerDist, ConvLayer):

    """
    Distributed convolutional layer.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, weight_init, activation=None):
        super(ConvLayerDist, self).__init__(name, backend, batch_size, pos,
                                            learning_rule, nifm, nofm,
                                            ifmshape, fshape, stride,
                                            activation=activation)
        self.nout = self.ofmsize * nofm
        self.weights = backend.gen_weights((self.fsize, nofm),
                                           weight_init)
        self.output = backend.zeros((batch_size, self.nout))
        self.updates = backend.zeros(self.weights.shape)
        self.prodbuf = backend.zeros((batch_size, nofm))
        self.bpropbuf = backend.zeros((batch_size, self.fsize))
        self.updatebuf = backend.zeros((self.fsize, nofm))
        self.learning_rule.allocate_state(self.updates)
        if activation is not None:
            raise NotImplementedError('TODO')

    def adjust_for_dist(self):
        self.ifmshape = self.input.local_array.ifmshape
        super(ConvLayerDist, self).adjust_for_dist(self.ifmshape)
        self.nout = self.ofmsize * self.nofm
        self.output = self.backend.zeros((self.batch_size, self.nout))

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(ConvLayerDist, self).fprop(inputs)

    def bprop(self, error, inputs, epoch):
        if self.pos > 0:
            self.backend.bprop_conv(self.weights, error, self.berror,
                                    self.links, self.ifmshape, self.ofmshape,
                                    self.ofmlocs, 0, self.stride, self.nifm,
                                    1, self.bpropbuf)
        self.backend.update_conv(self.weights, inputs, error, self.updates,
                                 self.links, self.ifmshape, self.ofmshape,
                                 self.ofmlocs, 0, self.stride, self.nifm,
                                 1, self.fwidth, self.updatebuf)

        # accumulate updates across tiles for all filters
        # if want to keep weights unshared across nodes, could not do the
        # transfers here
        self.updates._tensor = MPI.COMM_WORLD.reduce(
            self.updates.raw(), op=MPI.SUM, root=0)
        self.updates._tensor = MPI.COMM_WORLD.bcast(self.updates.raw())

        # Update the filters after summing the weight updates.
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class LocalFilteringLayer(LocalLayer):

    """
    Local filtering layer. This is very similar to ConvLayer, but the weights
    are not shared.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule,
                 nifm, nofm, ifmshape, fshape, stride, weight_init,
                 pretraining, sparsity, tied_weights):
        super(LocalFilteringLayer, self).__init__(name, backend, batch_size,
                                                  pos, learning_rule,
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
        self.learning_rule = learning_rule

        self.learning_rule.allocate_state(self.updates)
        if pretraining is True:
            self.sparsity = sparsity
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
        self.learning_rule.set_pretrain_mode(True)
        self.pooling = pooling
        self.defilter = LocalDeFilteringLayer(self, self.tied_weights)

    def train_mode(self):
        self.learning_rule.set_pretrain_mode(False)

    def pretrain(self, inputs, cost, epoch):
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
        self.defilter.bprop(error, self.output, epoch)
        # Now backward propagate the gradient of the output of the
        # pooling layer.
        error = ((self.sparsity / inputs.shape[0]) *
                 (self.backend.ones(self.pooling.output.shape)))
        self.pooling.bprop(error, self.output, epoch)
        # Aggregate the errors from both layers before back propagating
        # through the current layer.
        berror = self.defilter.berror + self.pooling.berror
        self.bprop(berror, inputs, epoch)
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
                             self.weights.take(self.ofmlocs[dst],
                                               axis=0).transpose(),
                             out=self.prodbuf)
            # size: # mbs x nofm
            self.output[:, self.ofmlocs[dst]] = self.prodbuf

    def bprop(self, error, inputs, epoch):
        self.delta = error
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in xrange(self.ofmsize):
                # Use the same filter that was used for forward propagation
                # of this receptive field.
                # size-guide
                # self.delta.take: # mbs x nofm
                # self.weights.take: # (nofm x fsize )
                self.backend.dot(self.delta.take(self.ofmlocs[dst], axis=1),
                                 self.weights.take(self.ofmlocs[dst], axis=0),
                                 self.bpropbuf)
                rflinks = self.rlinks[dst]
                self.backend.add(self.bpropbuf,
                                 self.berror.take(rflinks, axis=1),
                                 out=self.bpropbuf)
                self.berror[:, rflinks] = self.bpropbuf

        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            delta_slice = self.delta.take(self.ofmlocs[dst], axis=1)
            self.backend.dot(delta_slice.transpose(),
                             inputs.take(rflinks, axis=1),
                             out=self.updatebuf)
            self.updates[self.ofmlocs[dst]] = self.updatebuf

        self.learning_rule.apply_rule(self.weights, self.updates, epoch)
        self.normalize_weights(self.weights)


class LocalFilteringLayerDist(LocalLayerDist, LocalFilteringLayer):

    """
    Local filtering layer. This is very similar to ConvLayer, but the weights
    are not shared.
    """

    def adjust_for_dist(self):
        # shape with halos
        ifmshape = self.input.local_array.ifmshape
        top_left_row_output = self.input.local_array.top_left_row_output
        top_left_col_output = self.input.local_array.top_left_col_output

        super(LocalFilteringLayerDist, self).adjust_for_dist(ifmshape)
        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nout = self.ofmsize * self.nofm

        # for defiltering layer
        self.autoencoder = LocalArray(
            batch_size=self.batch_size,
            global_row_index=self.input.local_array.global_row_index,
            global_col_index=self.input.local_array.global_col_index,
            height=self.input.local_array.height,
            width=self.input.local_array.width,
            act_channels=self.input.local_array.act_channels,
            top_left_row=self.input.local_array.top_left_row,
            top_left_col=self.input.local_array.top_left_col,
            border_id=self.input.local_array.border_id,
            hsr_north=self.input.local_array.hsr_north,
            hsr_south=self.input.local_array.hsr_south,
            hsc_west=self.input.local_array.hsc_west,
            hsc_east=self.input.local_array.hsc_east,
            comm_per_dim=self.input.local_array.comm_per_dim,
            backend=self.backend)
        # reuse halo info from filtering layer
        self.autoencoder.send_halos = self.input.local_array.send_halos
        self.autoencoder.recv_halos = self.input.local_array.recv_halos
        self.autoencoder.local_image_indices = (
            self.input.local_array.local_image_indices)

        self.output = self.backend.zeros((self.batch_size, self.nout))

        # if initializing the weights from scratch
        # self.weights = self.backend.gen_weights((self.nout, self.fsize),
        #                                        self.weight_init, dtype=dtype)

        # if initializing using same seed as non-dist version
        # adjust size of self.weights for halo dimensions
        out_indices = []
        for cur_channel in range(self.nofm):
            current_index = (cur_channel * self.global_ofmsize +
                             top_left_row_output * self.global_ofmwidth +
                             top_left_col_output)
            for cur_row in range(self.ofmheight):
                out_indices.extend(
                    range(current_index, current_index + self.ofmwidth))
                current_index += self.global_ofmwidth
        self.weights = self.weights.take(out_indices, axis=0)

        self.normalize_weights(self.weights)
        self.updates = self.backend.zeros(self.weights.shape)
        self.learning_rule.allocate_state(self.updates)
        self.prodbuf = self.backend.zeros((self.batch_size, self.nofm))
        self.bpropbuf = self.backend.zeros((self.batch_size, self.fsize))
        self.updatebuf = self.backend.zeros((self.nofm, self.fsize))

    def __init__(self, name, backend, batch_size, pos, learning_rule,
                 nifm, nofm, ifmshape, fshape, stride, weight_init,
                 pretraining, sparsity, tied_weights):
        super(
            LocalFilteringLayerDist, self).__init__(name, backend, batch_size,
                                                    pos, learning_rule,
                                                    nifm, nofm, ifmshape,
                                                    fshape, stride)
        self.nout = self.ofmsize * nofm
        self.weight_init = weight_init
        self.weights = self.backend.gen_weights((self.nout, self.fsize),
                                                self.weight_init,
                                                dtype='float32')
        if pretraining is True:
            self.sparsity = sparsity
            self.tied_weights = tied_weights

    def pretrain_mode(self, pooling):
        super(LocalFilteringLayerDist, self).pretrain_mode(pooling)
        # temp1 stores a temp buffer without the chunk
        self.defilter.temp1 = [self.backend.zeros(
            (self.batch_size, self.input.local_array.local_array_size))]
        self.learning_rule.set_pretrain_mode(True)

    def pretrain(self, inputs_, cost, epoch):
        # Forward propagate the input through this layer and a
        # defiltering layer to reconstruct the input.
        inputs = self.fprop(inputs_)
        self.defilter.fprop(self.output)

        self.learning_rule.set_pretrain_mode(True)

        # halo aggregation across chunks for defiltering layer
        self.autoencoder.make_bprop_view(self.defilter.output)
        self.autoencoder.make_fprop_view(
            self.autoencoder.defiltering_local_image)

        # Forward propagate the output of this layer through a
        # pooling layer. The output of the pooling layer is used
        # to optimize sparsity.
        self.pooling.fprop(self.output)
        # Backward propagate the gradient of the reconstruction error
        # through the defiltering layer.
        error = cost.apply_derivative(self.backend,
                                      self.autoencoder.chunk,
                                      inputs,
                                      self.defilter.temp)
        self.backend.divide(error, self.backend.wrap(inputs.shape[0]),
                            out=error)
        self.defilter.bprop(error, self.output, epoch)
        # Now backward propagate the gradient of the output of the
        # pooling layer.
        error = ((self.sparsity / inputs.shape[0]) *
                 (self.backend.ones(self.pooling.output.shape)))
        self.pooling.bprop(error, self.output, epoch)
        berror = self.defilter.berror + (
            self.pooling.input.get_bprop_view(self.pooling.berror))
        self.bprop(berror, inputs, epoch)
        rcost = cost.apply_function(self.backend,
                                    self.autoencoder.defiltering_local_image,
                                    inputs_,
                                    self.defilter.temp1)
        spcost = self.sparsity * self.pooling.output.sum()
        return rcost, spcost

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(LocalFilteringLayerDist, self).fprop(inputs)
        return inputs


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
        self.learning_rule = prev.learning_rule
        self.learning_rule.set_pretrain_mode(True)
        self.backend = prev.backend
        self.rlinks = prev.rlinks
        self.prev = prev

    def fprop(self, inputs):
        self.backend.clear(self.output)
        for dst in xrange(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            # size guide:
            # inputs[:, self.prev.ofmlocs[dst]]: mbs x nout -> mbs x nofm
            # self.weights.take: nofm x ifmsize
            self.backend.dot(inputs[:, self.prev.ofmlocs[dst]],
                             self.weights.take(self.prev.ofmlocs[dst],
                                               axis=0),
                             out=self.prodbuf)
            self.output[:, rflinks] += self.prodbuf

    def bprop(self, error, inputs, epoch):
        for dst in xrange(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(error[:, rflinks],
                             self.weights.take(self.prev.ofmlocs[dst],
                                               axis=0).transpose(),
                             out=self.bpropbuf)
            self.berror[:, self.prev.ofmlocs[dst]] = self.bpropbuf
            delta_slice = error[:, rflinks]
            self.backend.dot(inputs[:, self.prev.ofmlocs[dst]].transpose(),
                             delta_slice,
                             out=self.updatebuf)
            self.updates[self.prev.ofmlocs[dst]] = self.updatebuf

        self.learning_rule.apply_rule(self.weights, self.updates, epoch)

        self.prev.normalize_weights(self.weights)


class MaxPoolingLayer(LocalLayer):

    """
    Max pooling layer.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(MaxPoolingLayer, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm, ifmshape,
            fshape, stride, pooling=True)
        self.maxinds = backend.alloc(batch_size * nifm, self.ofmsize,
                                     dtype='i32')
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.alloc(self.batch_size, self.nout)

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
        self.backend.fprop_mpool(
            inputs, self.output, self.links,
            self.ifmshape, self.ofmshape, self.fshape, 0,
            self.stride, self.nifm, self.maxinds)

    def bprop(self, error, inputs, epoch):
        if self.pos > 0:
            self.backend.bprop_mpool(
                inputs, self.output,
                error, self.berror, self.links, self.ifmshape, self.ofmshape,
                self.fshape, 0, self.stride, self.nifm, self.maxinds)


class MaxPoolingLayerDist(LocalLayerDist, MaxPoolingLayer):

    """
    Distributed Max pooling layer.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(MaxPoolingLayerDist, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm, ifmshape,
            fshape, stride, pooling=True)
        self.maxinds = backend.alloc(batch_size * nifm, self.ofmsize,
                                     dtype='i32')
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.alloc(self.batch_size, self.nout)

    def adjust_for_dist(self):
        self.ifmshape = self.input.local_array.ifmshape
        super(MaxPoolingLayerDist, self).adjust_for_dist(self.ifmshape)
        self.prodbuf = self.backend.zeros(
            (self.batch_size * self.nifm, self.fshape[0] * self.fshape[1]))

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(MaxPoolingLayerDist, self).fprop(inputs)


class L2PoolingLayer(LocalLayer):

    """
    L2 pooling layer. Each receptive field is pooled to obtain its L2 norm
    as output.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(L2PoolingLayer, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True)
        self.prodbuf = self.backend.zeros((batch_size * nifm,
                                           self.fshape[0] * self.fshape[1]))
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.alloc(self.batch_size, self.nout)

    def __str__(self):
        return ("L2PoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        self.backend.fprop_l2pool(
            inputs, self.output, self.links,
            self.ifmshape, self.ofmshape, self.fshape,
            0, self.stride, self.nifm)

    def bprop(self, error, inputs, epoch):
        if self.pos > 0:
            self.backend.bprop_l2pool(
                inputs, self.output, error, self.berror, self.links,
                self.ifmshape, self.ofmshape, self.fshape,
                0, self.stride, self.nifm, self.prodbuf)


class L2PoolingLayerDist(LocalLayerDist, L2PoolingLayer):

    """
    Distributed L2 pooling layer. Each receptive field is pooled to obtain its
    L2 norm as output.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(L2PoolingLayerDist, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True)
        self.prodbuf = self.backend.zeros((batch_size * nifm,
                                           self.fshape[0] * self.fshape[1]))
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.alloc(self.batch_size, self.nout)

    def adjust_for_dist(self):
        # shape with halos
        ifmshape = self.input.local_array.ifmshape
        super(L2PoolingLayerDist, self).adjust_for_dist(ifmshape)
        self.prodbuf = self.backend.zeros(
            (self.batch_size * self.nifm, self.fshape[0] * self.fshape[1]))

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(L2PoolingLayerDist, self).fprop(inputs)

    def bprop(self, error, inputs_, epoch):
        # redo-ing get_fprop_view, could cache for speed-up
        inputs = self.input.get_fprop_view(inputs_)
        super(L2PoolingLayerDist, self).bprop(error, inputs, epoch)


class AveragePoolingLayer(LocalLayer):

    """
    Average pooling.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(AveragePoolingLayer, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True)
        self.nout = nifm * self.ofmsize
        self.output = self.backend.alloc(self.batch_size, self.nout)

    def __str__(self):
        return ("AveragePoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        self.backend.fprop_apool(
            inputs, self.output, self.links,
            self.ifmshape, self.ofmshape, self.fshape,
            0, self.stride, self.nifm)

    def bprop(self, error, inputs, epoch):
        if self.pos > 0:
            self.backend.bprop_apool(
                self.output, error, self.berror, self.links,
                self.ifmshape, self.ofmshape, self.fshape,
                0, self.stride, self.nifm)


class AveragePoolingLayerDist(LocalLayerDist, AveragePoolingLayer):

    """
    Distributed Average pooling layer.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(AveragePoolingLayerDist, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True)
        self.prodbuf = self.backend.zeros((batch_size * nifm,
                                           self.fshape[0] * self.fshape[1]))
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.alloc(self.batch_size, self.nout)

    def adjust_for_dist(self):
        # shape with halos
        ifmshape = self.input.local_array.ifmshape
        super(AveragePoolingLayerDist, self).adjust_for_dist(ifmshape)
        self.prodbuf = self.backend.zeros(
            (self.batch_size * self.nifm, self.fshape[0] * self.fshape[1]))

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(AveragePoolingLayerDist, self).fprop(inputs)

    def bprop(self, error, inputs_, epoch):
        # redo-ing get_fprop_view, could cache for speed-up
        inputs = self.input.get_fprop_view(inputs_)
        super(AveragePoolingLayerDist, self).bprop(error, inputs, epoch)


class Convolver(LocalLayer):

    """
    Lightweight convolutional layer that only does fprop.
    """

    def __init__(self, backend, batch_size, nifm,
                 nofm, ifmshape, fshape, stride, weights):
        super(Convolver, self).__init__('conv', backend, batch_size, 0,
                                        0.0, nifm, nofm,
                                        ifmshape, fshape, stride)
        self.nout = self.ofmsize * nofm
        self.weights = weights
        self.output = backend.zeros((batch_size, self.nout))
        self.prodbuf = backend.zeros((batch_size, nofm))

    def fprop(self, inputs):
        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(inputs.take(rflinks, axis=1),
                             self.weights.transpose(), out=self.prodbuf)
            self.output[:, self.ofmlocs[dst]] = self.prodbuf


class LCNLayer(YAMLable):

    """
    Local contrast normalization.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        self.name = name
        self.backend = backend
        self.ifmshape = ifmshape
        self.ifmheight, self.ifmwidth = ifmshape
        self.fheight, self.fwidth = fshape
        self.fsize = nifm * self.fheight * self.fwidth
        self.batch_size = batch_size
        self.nifm = nifm
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.nin = nifm * self.ifmsize
        self.nout = self.nin

        self.filters = self.normalized_gaussian_filters(nifm, fshape)
        # self.fpeakdiff = 1.0 - self.fpeak
        self.stride = stride
        self.fshape = fshape
        self.pos = pos

        self.exifmheight = (self.ifmheight - 1) * stride + self.fheight
        self.exifmwidth = (self.ifmwidth - 1) * stride + self.fwidth
        self.exifmsize = self.exifmheight * self.exifmwidth
        self.exifmshape = (self.exifmheight, self.exifmwidth)

        self.exinputs = self.backend.zeros((batch_size, nifm * self.exifmsize))
        self.rexinputs = self.exinputs.reshape((self.batch_size, self.nifm,
                                                self.exifmheight,
                                                self.exifmwidth))
        self.conv = Convolver(backend, batch_size, nifm, 1,
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
        self.routput = self.output.reshape((batch_size, nifm,
                                            self.ifmheight,
                                            self.ifmwidth))
        self.subout = backend.zeros(self.output.shape)
        self.rsubout = self.subout.reshape(self.routput.shape)
        self.subtemp = backend.zeros(self.output.shape)
        self.rsubtemp = self.subtemp.reshape(self.routput.shape)
        if pos > 0:
            self.diverror = backend.zeros((batch_size, self.nin))
            self.exerror = self.backend.zeros((batch_size,
                                               nifm * self.exifmsize))
            self.rexerror = self.exerror.reshape((batch_size, nifm,
                                                  self.exifmheight,
                                                  self.exifmwidth))
            self.prodbuf = self.backend.zeros((batch_size, self.fsize))
            self.bprop_filters = self.backend.zeros((nifm,
                                                     self.filters.shape[0],
                                                     self.filters.shape[1]))
            self.sqtemp = backend.zeros(self.output.shape)
            for fm in xrange(nifm):
                self.bprop_filters[fm] = self.filters.copy()
                rfilter = self.bprop_filters[fm].reshape(
                    (nifm, self.fheight, self.fwidth))
                rfilter[fm, self.fheight / 2, self.fwidth / 2] -= 1.0

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
        assert(len(shape) == 2)
        single = gaussian_filter(shape)
        single /= (count * single.sum())
        assert shape[0] % 2 == 1
        assert shape[1] % 2 == 1
        filters = self.backend.zeros((count, shape[0], shape[1]))
        filters[:] = single

        filters = filters.reshape((1, count * shape[0] * shape[1]))
        return filters

    def copy_to_inset(self, canvas, inset, start_row, start_col):
        canvas[:, :, start_row:(canvas.shape[2] - start_row),
               start_col:(canvas.shape[3] - start_col)] = inset

    def copy_from_inset(self, canvas, start_row, start_col):
        return canvas[:, :, self.start_row:(canvas.shape[2] - start_row),
                      self.start_col:(canvas.shape[3] - start_col)]

    def fprop_sub_normalize(self, inputs):
        rinputs = inputs.reshape((self.batch_size, self.nifm,
                                  self.ifmheight, self.ifmwidth))
        self.copy_to_inset(self.rexinputs, rinputs,
                           self.start_row, self.start_col)
        # Convolve with gaussian filters to obtain a "mean" feature map.
        self.conv.fprop(self.exinputs)
        self.backend.subtract(rinputs, self.rmeanfm, out=self.rsubout)

    def fprop_div_normalize(self):
        self.backend.multiply(self.subout, self.subout, out=self.subtemp)
        self.copy_to_inset(self.rexinputs, self.rsubtemp,
                           self.start_row, self.start_col)

        self.conv.fprop(self.exinputs)
        self.backend.sqrt(self.meanfm, out=self.meanfm)
        assert self.subout[self.meanfm.raw() == 0.0].sum() == 0.0
        self.meanfm[self.meanfm.raw() == 0.0] = 1.0
        self.backend.divide(self.rsubout, self.rmeanfm, out=self.routput)

    def fprop(self, inputs):
        self.backend.clear(self.exinputs)
        self.fprop_sub_normalize(inputs)
        self.fprop_div_normalize()

    def reshape_error(self):
        # discards zero padding around the delta matrix
        self.berror = self.copy_from_inset(self.rexerror, self.start_row,
                                           self.start_col)
        self.berror = self.berror.reshape((self.batch_size, self.nin))

    def bprop_sub_normalize(self, error, inputs, epoch):
        self.backend.clear(self.exerror)
        for fm in range(self.nifm):
            for dst in xrange(self.conv.ofmsize):
                rflinks = self.conv.rlinks[dst]
                loc = self.conv.ofmlocs[dst].raw() + self.conv.ofmsize * fm
                filt = self.bprop_filters[fm]
                self.backend.multiply(error[:, loc], filt, out=self.prodbuf)
                self.exerror[:, rflinks] -= self.prodbuf
        self.reshape_error()

    def bprop_div_normalize(self, error, inputs, epoch):
        self.backend.clear(self.exerror)
        self.backend.cube(self.output, out=self.diverror)
        self.subtemp[:] = self.subout
        assert self.diverror[self.subout.raw() == 0].sum() == 0.0
        self.subout[self.subout.raw() == 0] = 1.0
        self.backend.square(self.subout, out=self.sqtemp)
        # this is for the non-padded, non-halo matrix only
        self.backend.divide(self.diverror, self.sqtemp, out=self.diverror)

        for fm in range(self.nifm):
            for dst in xrange(self.conv.ofmsize):
                # self.conv.ofmlocs is over 1 fm only
                loc = self.conv.ofmlocs[dst].raw() + self.conv.ofmsize * fm
                divout = self.output.take(loc, axis=1)
                subout = self.subout.take(loc, axis=1)
                assert divout[subout.raw() == 0].sum() == 0
                subout[subout.raw() == 0.0] = 1.0
                self.backend.divide(divout, subout, out=divout)

                rflinks = self.conv.rlinks[dst]
                self.copy_to_inset(self.rexinputs, self.rsubtemp,
                                   self.start_row, self.start_col)
                rrexinputs = self.rexinputs.reshape(
                    (self.batch_size, self.nifm * self.exifmsize))
                frame = rrexinputs.take(rflinks, axis=1)
                self.backend.multiply(frame, self.filters, out=frame)
                self.backend.multiply(frame, self.diverror[:, loc], out=frame)
                rframe = frame.reshape((self.batch_size, self.nifm,
                                        self.fheight, self.fwidth))
                # this is working on the g2/y2 term
                rframe[:, fm:(fm + 1),
                       self.fheight / 2, self.fwidth / 2] -= divout
                self.backend.multiply(error[:, loc].repeat(self.fsize, axis=1),
                                      frame, out=frame)
                self.exerror[:, rflinks] -= frame
        self.reshape_error()

    def bprop(self, error, inputs, epoch):
        if self.pos > 0:
            # note: have to account for halos + padding after each step
            self.bprop_div_normalize(error, inputs, epoch)
            self.bprop_sub_normalize(self.berror, inputs, epoch)

    def bprop_fast(self, error, inputs, epoch):
        """
        An incorrect, but much faster version of backprop.
        """
        if self.pos > 0:
            self.berror[:] = error


class LCNLayerDist(LCNLayer):

    """
    Distributed Local contrast normalization.
    """

    def adjust_for_dist(self):
        # output dims are same as input dims (w/o halo) for LCN layer
        output_height = self.input.local_array.height
        output_width = self.input.local_array.width
        # shape with halos
        ifmshape = self.input.local_array.ifmshape
        border_id = self.input.border_id

        self.ifmshape = ifmshape
        self.ifmheight, self.ifmwidth = ifmshape  # with halos, but not padding
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.nin = self.nifm * self.ifmsize
        self.nout = output_height * output_width * self.nifm
        self.filters = self.normalized_gaussian_filters(
            self.nifm, self.fshape)

        # if border_id != gc.CENTER:
        pad_height = self.fheight - 1
        pad_width = self.fwidth - 1

        # compute how much to pad
        pad_width_right = pad_width // 2
        pad_width_left = pad_width - pad_width_right
        pad_height_bottom = pad_height // 2
        pad_height_top = pad_height - pad_height_bottom

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
        if border_id in [gc.SINGLE]:
            top_padding = pad_height_top
            bottom_padding = pad_height_bottom
            left_padding = pad_width_left
            right_padding = pad_width_right
            self.start_row = top_padding
            self.start_col = left_padding

        # todo: only supports stride of 1 for now
        self.exifmheight = (self.ifmheight) * self.stride + (
            top_padding + bottom_padding)
        self.exifmwidth = (self.ifmwidth) * self.stride + (
            left_padding + right_padding)
        self.exifmsize = self.exifmheight * self.exifmwidth
        self.exifmshape = (self.exifmheight, self.exifmwidth)

        self.exinputs = self.backend.zeros((self.batch_size,
                                            self.nifm * self.exifmsize))
        self.rexinputs = self.exinputs.reshape((self.batch_size, self.nifm,
                                                self.exifmheight,
                                                self.exifmwidth))
        self.conv = Convolver(self.backend, self.batch_size, self.nifm, 1,
                              self.exifmshape, self.fshape, self.stride,
                              self.filters)
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

        self.output = self.backend.zeros((self.batch_size, self.nout))
        self.routput = self.output.reshape((self.batch_size, self.nifm,
                                            output_height, output_width))

        self.temp1 = self.backend.zeros(self.output.shape)
        self.rtemp1 = self.temp1.reshape(self.routput.shape)
        self.temp2 = self.backend.zeros(self.output.shape)
        self.rtemp2 = self.temp2.reshape(self.routput.shape)
        self.subout = self.backend.zeros(self.output.shape)
        self.rsubout = self.subout.reshape(self.routput.shape)
        self.subtemp = self.backend.zeros(self.output.shape)
        self.rsubtemp = self.subtemp.reshape(self.routput.shape)
        self.subtemp2 = self.backend.zeros((self.batch_size, self.nin))
        self.rsubtemp2 = self.subtemp2.reshape((self.batch_size, self.nifm,
                                                self.ifmheight, self.ifmwidth))

        if self.pos > 0:
            # changed to nout for bprop in dist version, compared to nin in
            # non-dist version
            self.diverror = self.backend.zeros(
                (self.batch_size, self.nout))
            self.exerror = self.backend.zeros((self.batch_size,
                                               self.nifm * self.exifmsize))
            self.rexerror = self.exerror.reshape((self.batch_size, self.nifm,
                                                  self.exifmheight,
                                                  self.exifmwidth))
            self.prodbuf = self.backend.zeros(
                (self.batch_size, self.fsize))
            self.bprop_filters = self.backend.zeros((self.nifm,
                                                     self.filters.shape[0],
                                                     self.filters.shape[1]))
            self.sqtemp = self.backend.zeros(self.output.shape)
            for fm in xrange(self.nifm):
                self.bprop_filters[fm] = self.filters.copy()
                rfilter = self.bprop_filters[fm].reshape(
                    (self.nifm, self.fheight, self.fwidth))
                rfilter[fm, self.fheight / 2, self.fwidth / 2] -= 1.0

    def copy_to_inset(self, canvas, inset, start_row, start_col):
        canvas[:, :, start_row:start_row + inset.shape[2],
               start_col:start_col + inset.shape[3]] = inset

    def copy_from_inset(self, canvas, start_row, start_col):
        return canvas[:, :, start_row:start_row + self.ifmheight,
                      start_col:start_col + self.ifmwidth]

    def fprop_sub_normalize(self, inputs):
        rinputs = inputs.reshape((self.batch_size, self.nifm,
                                  self.ifmheight, self.ifmwidth))
        self.copy_to_inset(self.rexinputs, rinputs,
                           self.start_row, self.start_col)
        # Convolve with gaussian filters to obtain a "mean" feature map.
        self.conv.fprop(self.exinputs)
        # rinputs includes halos but not padding
        self.backend.subtract(
            self.rexinputs[:, :,
                           self.start_row2:(
                               self.rexinputs.shape[2] - self.start_row2),
                           self.start_col2:(
                               self.rexinputs.shape[3] - self.start_col2)],
            self.rmeanfm,
            out=self.rsubout)

    def fprop_div_normalize(self):
        self.backend.multiply(self.input.local_array.chunk,
                              self.input.local_array.chunk,
                              out=self.subtemp2)
        self.copy_to_inset(self.rexinputs, self.rsubtemp2,
                           self.start_row, self.start_col)
        self.conv.fprop(self.exinputs)
        self.backend.sqrt(self.meanfm, out=self.meanfm)
        assert self.subout[self.meanfm.raw() == 0.0].sum() == 0.0
        self.meanfm[self.meanfm.raw() == 0.0] = 1.0
        self.backend.divide(
            self.input.get_local_acts().reshape(
                self.routput.shape), self.rmeanfm, out=self.routput)

    def fprop(self, inputs_):
        self.backend.clear(self.exinputs)
        inputs = self.input.get_fprop_view(inputs_)
        self.fprop_sub_normalize(inputs)
        # distributed version
        self.input.make_fprop_view(self.subout)
        self.fprop_div_normalize()

    def bprop_div_normalize(self, error, inputs, epoch):
        self.backend.clear(self.exerror)
        self.backend.cube(self.output, out=self.diverror)

        self.subout = self.input.get_local_acts()
        self.subtemp2[:] = self.input.local_array.chunk

        self.subtemp[:] = self.subout
        assert self.diverror[self.subout.raw() == 0].sum() == 0.0
        self.subout[self.subout.raw() == 0] = 1.0
        self.backend.square(self.subout, out=self.sqtemp)
        # this is for the non-padded, non-halo matrix only
        self.backend.divide(self.diverror, self.sqtemp, out=self.diverror)

        for fm in range(self.nifm):
            for dst in xrange(self.conv.ofmsize):
                # self.conv.ofmlocs is over 1 fm only
                loc = self.conv.ofmlocs[dst].raw() + self.conv.ofmsize * fm
                divout = self.output.take(loc, axis=1)
                subout = self.subout.take(loc, axis=1)
                assert divout[subout.raw() == 0].sum() == 0
                subout[subout.raw() == 0.0] = 1.0
                self.backend.divide(divout, subout, out=divout)

                rflinks = self.conv.rlinks[dst]
                self.copy_to_inset(self.rexinputs, self.rsubtemp2,
                                   self.start_row, self.start_col)
                rrexinputs = self.rexinputs.reshape(
                    (self.batch_size, self.nifm * self.exifmsize))
                frame = rrexinputs.take(rflinks, axis=1)
                self.backend.multiply(frame, self.filters, out=frame)
                self.backend.multiply(frame, self.diverror[:, loc], out=frame)
                rframe = frame.reshape((self.batch_size, self.nifm,
                                        self.fheight, self.fwidth))
                # this is working on the g2/y2 term
                rframe[:, fm:(fm + 1),
                       self.fheight / 2, self.fwidth / 2] -= divout
                self.backend.multiply(error[:, loc].repeat(self.fsize, axis=1),
                                      frame, out=frame)
                self.exerror[:, rflinks] -= frame
        self.reshape_error()

    def bprop(self, error, inputs, epoch):
        if self.pos > 0:
            # note: have to account for halos + padding after each step
            self.bprop_div_normalize(error, inputs, epoch)

            self.bprop_sub_normalize(
                self.input.get_bprop_view(self.berror),
                inputs, epoch)

            self.berror = (
                self.input.get_bprop_view(self.berror))
