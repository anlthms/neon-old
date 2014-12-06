# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
import numpy as np
from neon.backends.cpu import CPU
from neon.transforms.gaussian import gaussian_filter
from neon.util.compat import MPI_INSTALLED, range
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
                 weight_init, learning_rule, activation=None,
                 weight_dtype=None, updates_dtype=None, pre_act_dtype=None,
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
        self.updates = self.backend.empty(self.weights.shape, updates_dtype)
        self.updates_dtype = updates_dtype
        self.output = self.backend.zeros((self.nout, batch_size), output_dtype)
        if activation is not None:
            self.pre_act = self.backend.zeros((self.nout, batch_size),
                                              pre_act_dtype)
        else:
            self.pre_act = self.output
        self.pos = pos
        self.learning_rule = learning_rule
        self.learning_rule.allocate_state(self.updates)
        self.batch_size = batch_size
        if pos > 0:
            # This is storage for the backward propagated error.
            self.berror = self.backend.empty((nin - 1, batch_size),
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
        self.backend.fprop_fc(inputs, self.weights, out=self.pre_act)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs):
        """
        # numpy pseudocode for the backprop:
        # updates = dot(error.transpose(), inputs)  # calculate new gradient
        # weight update itself done by application of learning rule
        """
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)

        if self.pos > 0:
            endcol = self.weights.shape[1] - 1
            self.backend.bprop_fc(error, self.weights[:, 0:endcol],
                                  out=self.berror)

        inputs = self.backend.append_bias(inputs)
        self.backend.update_fc(error, inputs, out=self.updates)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)

    def set_train_mode(self, mode):
        pass


class LayerDist(Layer):

    def adjust_for_dist(self):
        # indices of the input layer in weight matrix
        in_indices = []
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
                    in_indices.extend(
                        range(current_index, current_index + self.ifmshape[1]))
                    current_index += self.global_width
        elif self.prev_layer == 'LayerDist':
            in_indices = self.in_indices
        else:
            raise ValueError('Unsupported previous layer for '
                             'LayerDist')

        self.weights = self.weights.take(in_indices, axis=1)

        self.updates = self.backend.empty(self.weights.shape)
        self.learning_rule.allocate_state(self.updates)
        self.delta_ = self.backend.empty((self.nout_, self.batch_size))
        self.delta_gather = self.backend.empty(
            (self.nout, self.batch_size * MPI.COMM_WORLD.size))
        if self.pos > 0:
            # This is storage for the backward propagated error.
            if MPI.COMM_WORLD.rank == MPI.COMM_WORLD.size - 1:
                self.berror = self.backend.empty((self.nin - 1,
                                                  self.batch_size))
            else:
                self.berror = self.backend.empty((self.nin, self.batch_size))

    def fprop(self, inputs):
        if MPI.COMM_WORLD.rank == MPI.COMM_WORLD.size - 1:
            inputs = self.backend.append_bias(inputs)
        self.backend.fprop_fc(inputs, self.weights, out=self.pre_act)
        # accumulate the pre_act values before applying non-linearity
        self.pre_act._tensor = MPI.COMM_WORLD.reduce(
            self.pre_act.raw(), op=MPI.SUM, root=0)
        # apply non-linearity on the output node
        if MPI.COMM_WORLD.rank == 0 and self.activation is not None:
            # this stores the derivatives in self.pre_act
            self.activation.apply_both(self.backend, self.pre_act, self.output)
        # strictly, following line not needed for top-most layer
        self.output._tensor = MPI.COMM_WORLD.bcast(self.output.raw())
        # broadcast back the pre_act values for bprop.
        # note: suboptimal for dist implementation,
        # but a consequence of reusing the pre_act buffer for fprop and bprop
        self.pre_act._tensor = MPI.COMM_WORLD.bcast(self.pre_act.raw())

    def bprop(self, error, inputs):
        """
        # numpy pseudocode for the backprop:
        # updates  = dot(error.T, inputs)        # calculate new gradient
        # weight update itself done by application of learning rule
        """
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act_, out=error)
        endcol = self.weights.shape[1]
        if MPI.COMM_WORLD.rank == MPI.COMM_WORLD.size - 1:
            inputs = self.backend.append_bias(inputs)
            endcol = self.weights.shape[1] - 1
        if self.nout_ != self.nout:
            MPI.COMM_WORLD.Allgather(
                error.raw(), self.delta_gather._tensor)
            # todo: only supported in numpy backend for now
            self.delta_._tensor = np.hstack(
                np.split(self.delta_gather.raw(), MPI.COMM_WORLD.size))
            if self.pos > 0:
                self.backend.bprop_fc(self.delta_,
                                      self.weights[:, 0:endcol],
                                      out=self.berror)
            self.backend.update_fc(self.delta_, inputs, out=self.updates)
        else:
            if self.pos > 0:
                self.backend.bprop_fc(error,
                                      self.weights[:, 0:endcol],
                                      out=self.berror)
            self.backend.update_fc(error, inputs, out=self.updates)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class LayerWithNoBias(Layer):

    """
    Single NNet layer with no bias node
    """

    def __init__(self, name, backend, batch_size, pos, nin, nout,
                 weight_init, learning_rule, activation=None,
                 weight_dtype=None, updates_dtype=None, pre_act_dtype=None,
                 output_dtype=None, berror_dtype=None):
        super(LayerWithNoBias, self).__init__(name, backend, batch_size,
                                              pos, nin, nout, weight_init,
                                              learning_rule, activation)
        if pos > 0:
            self.berror = backend.empty((nin, batch_size))

    def fprop(self, inputs):
        self.backend.fprop_fc(inputs, self.weights, out=self.pre_act)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs):
        # comment if not using denominator term in cross_entropy
        self.backend.multiply(error, self.pre_act, out=error)
        if self.pos > 0:
            self.backend.bprop_fc(error, self.weights, out=self.berror)
        self.backend.update_fc(error, inputs, out=self.updates)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class RecurrentOutputLayer(Layer):

    """
    Derived from LayerWithNoBias. pre_act becomes pre_act_list, output becomes
    output_list, which are indexed by [tau], the unrolling step.
    """

    def __init__(self, name, backend, batch_size, pos, nin, nout, unrolls,
                 activation, weight_init, learning_rule, weight_dtype=None,
                 delta_dtype=None, updates_dtype=None, pre_act_dtype=None,
                 output_dtype=None, berror_dtype=None):
        super(RecurrentOutputLayer, self).__init__(name, backend, batch_size,
                                                   pos, nin, nout, weight_init,
                                                   learning_rule, activation)
        self.pre_act_list = [self.backend.zeros((batch_size, self.nout),
                                                pre_act_dtype)
                             for k in range(unrolls)]
        self.output_list = [self.backend.zeros((batch_size, self.nout),
                                               output_dtype)
                            for k in range(unrolls)]
        self.temp_out = self.backend.zeros((self.nout, self.nin))
        self.deltas_o = [self.backend.zeros((self.batch_size, nout))
                         for k in range(unrolls + 1)]
        if pos > 0:
            self.berror = backend.zeros((batch_size, nin))

    def fprop(self, inputs, tau):
        self.backend.fprop_fc(self.weights.transpose(), inputs,
                              out=self.pre_act_list[tau])
        self.activation.apply_both(self.backend,
                                   self.pre_act_list[tau],
                                   self.output_list[tau])

    def bprop(self, error, inputs, tau):
        self.deltas_o[tau] = error * self.pre_act_list[tau - 1]
        self.backend.update_fc(self.deltas_o[tau].transpose(),
                               inputs.transpose(), out=self.temp_out)
        self.updates += self.temp_out

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class RecurrentLSMTLayer(Layer):

    """
    Hidden layer with LSTM gates.
    """

    def __init__(self, name, backend, batch_size, pos, nin, nout, unrolls,
                 activation, weight_init, weight_init_rec, learning_rule,
                 weight_dtype=None, delta_dtype=None, updates_dtype=None,
                 pre_act_dtype=None, output_dtype=None, berror_dtype=None):
        # super calls into Layer.__init__() for weight init.
        super(RecurrentHiddenLayer, self).__init__(name, backend, batch_size,
                                                   pos, nin, nout, weight_init,
                                                   learning_rule, activation)
        # create weight matrices
        self.Wxi = self.backend.gen_weights((nout, nout),
                                            weight_init_rec,
                                            weight_dtype)
        self.Whi = self.backend.gen_weights((nout, nout),
                                            weight_init_rec,
                                            weight_dtype)
        self.Wxf = self.backend.gen_weights((nout, nout),
                                            weight_init_rec,
                                            weight_dtype)
        self.Whf = self.backend.gen_weights((nout, nout),
                                            weight_init_rec,
                                            weight_dtype)
        self.Wxo = self.backend.gen_weights((nout, nout),
                                            weight_init_rec,
                                            weight_dtype)
        self.Who = self.backend.gen_weights((nout, nout),
                                            weight_init_rec,
                                            weight_dtype)
        self.Wxc = self.backend.gen_weights((nout, nout),
                                            weight_init_rec,
                                            weight_dtype)
        self.Whc = self.backend.gen_weights((nout, nout),
                                            weight_init_rec,
                                            weight_dtype)

        # initialize buffers for intermediate values
        self.i_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.f_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.o_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.g_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.c_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.h_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.i_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]

        # preactivation -- do we really need to store this across unrolls?
        self.net_ix = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_ih = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_fx = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_fh = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_ox = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_oh = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_gx = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_gh = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]

        # misc
        self.deltas = [self.backend.zeros((self.batch_size, nout))
                       for k in range(unrolls + 1)]
        self.updates_rec = self.backend.zeros((self.nout, self.nout))
        self.temp_rec = self.backend.zeros((self.nout, self.nout))
        self.temp_in = self.backend.zeros((self.nout, self.nin))
        self.learning_rule.allocate_state_rec(self.updates_rec)

        self.berror = backend.zeros((batch_size, nout))

    def fprop(self, y, inputs, tau):
        """
        In numpy pseudocode, the forward pass is:
            de_dW = dot((y-t) / (y * (1-y)), dy_dW)     # d/dW CE(y,t)
            dy_dW = dot(sig_prime(dot(Wyh, h)), dh_dW)  # d/dW sigm(Wyh*h)
            dh_dW = o .* dp_dW + tanh(c) .* do_dW       # d/dW o .* tanh(c)
            do_dWxo = sigmoid_prime(dot(Wxo, x) +
                                    dot(Who, h) + b) x  # d/dW s(Wcx*x+Wch*h+b)
            dp_dW = dot(tanh_prime(c), dc_dW)           # d/dW phi(c)
            dc_dW = c_.*df_dW + i.*dg_dW + g .* di_dW   # d/dW (f.*c_ + i.*g)
            df_dWxf = sigmoid_prime(dot(Wxf, x) +
                                    dot(Whf, h) + b) x  # d/dW s(Wfx*x+Wfh*h+b)
            dg_dWxc = sigmoid_prime(dot(Wxc, x) +
                                    dot(Whc, h) + b) x  # d/dW s(Wcx*x+Wch*h+b)
            di_dWxi = sigmoid_prime(dot(Wxi, x) +
                                    dot(Whi, h) + b) x  # d/dW s(Wix*x+Wih*h+b)
        Start by computing the sigmoids and go up from there.
        This is for the d/dWx, but d/dWh follows the same schema.
        """
        batch_size = self.batch_size
        unrolls = self.unrolls

        self.net_ix = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_ih = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_fx = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_fh = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_ox = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_oh = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_gx = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]
        self.net_gh = [self.backend.zeros((batch_size, self.nout))
                       for k in range(unrolls)]

        self.i_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.f_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.o_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.g_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.c_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.h_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]
        self.i_t = [self.backend.zeros((batch_size, self.nout))
                    for k in range(unrolls)]

        # old code from RNN
        z1 = self.backend.zeros(self.pre_act_list[tau].shape)
        z2 = self.backend.zeros(self.pre_act_list[tau].shape)
        self.backend.fprop_fc(self.weights_rec.transpose(), y, out=z1)
        self.backend.fprop_fc(self.weights.transpose(), inputs, out=z2)
        self.pre_act_list[tau] = z1 + z2
        self.activation.apply_both(self.backend,
                                   self.pre_act_list[tau],
                                   self.output_list[tau])

    def bprop(self, error, inputs, tau, batch_inx):
        self.deltas[1] = error * self.pre_act_list[tau - 1]
        self.backend.update_fc(self.deltas[1].transpose(),
                               inputs[batch_inx[:, tau - 1]].transpose(),
                               out=self.temp_in)
        self.updates += self.temp_in
        for layer in range(0, tau - 1)[::-1]:
            self.backend.bprop_fc(self.weights_rec,
                                  self.deltas[tau - layer - 1].transpose(),
                                  out=self.berror)
            self.deltas[tau - layer] = self.berror * self.pre_act_list[layer]
            self.backend.update_fc(self.deltas[tau - (layer + 1)].transpose(),
                                   self.output_list[layer].transpose(),
                                   out=self.temp_rec)
            self.updates_rec += self.temp_rec
            self.backend.update_fc(self.deltas[tau - layer].transpose(),
                                   inputs[batch_inx[:, layer]].transpose(),
                                   out=self.temp_in)
            self.updates += self.temp_in

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)
        self.learning_rule.apply_rule_rec(self.weights_rec,
                                          self.updates_rec, epoch)


class RecurrentHiddenLayer(Layer):

    """
    Derived from LayerWithNoBias. In addition to the lists[tau] outlined for
    RecurrentOutputLayer, the fprop is getting input from two weight matrices,
    one connected to the input and one connected to the previous hidden state.
    """

    def __init__(self, name, backend, batch_size, pos, nin, nout, unrolls,
                 activation, weight_init, weight_init_rec, learning_rule,
                 weight_dtype=None, delta_dtype=None, updates_dtype=None,
                 pre_act_dtype=None, output_dtype=None, berror_dtype=None):
        # super calls into Layer.__init__() for weight init.
        super(RecurrentHiddenLayer, self).__init__(name, backend, batch_size,
                                                   pos, nin, nout, weight_init,
                                                   learning_rule, activation)
        self.weights_rec = self.backend.gen_weights((nout, nout),
                                                    weight_init_rec,
                                                    weight_dtype)
        self.pre_act_list = [self.backend.zeros((batch_size, self.nout),
                                                pre_act_dtype)
                             for k in range(unrolls)]
        self.output_list = [self.backend.zeros((batch_size, self.nout),
                                               output_dtype)
                            for k in range(unrolls)]
        self.deltas = [self.backend.zeros((self.batch_size, nout))
                       for k in range(unrolls + 1)]
        self.updates_rec = self.backend.zeros((self.nout, self.nout))
        self.temp_rec = self.backend.zeros((self.nout, self.nout))
        self.temp_in = self.backend.zeros((self.nout, self.nin))
        self.learning_rule.allocate_state_rec(self.updates_rec)

        self.berror = backend.zeros((batch_size, nout))

    def fprop(self, y, inputs, tau):
        z1 = self.backend.zeros(self.pre_act_list[tau].shape)
        z2 = self.backend.zeros(self.pre_act_list[tau].shape)
        self.backend.fprop_fc(self.weights_rec.transpose(), y, out=z1)
        self.backend.fprop_fc(self.weights.transpose(), inputs, out=z2)
        self.pre_act_list[tau] = z1 + z2
        self.activation.apply_both(self.backend,
                                   self.pre_act_list[tau],
                                   self.output_list[tau])

    def bprop(self, error, inputs, tau, batch_inx):
        self.deltas[1] = error * self.pre_act_list[tau - 1]
        self.backend.update_fc(self.deltas[1].transpose(),
                               inputs[batch_inx[:, tau - 1]].transpose(),
                               out=self.temp_in)
        self.updates += self.temp_in
        for layer in range(0, tau - 1)[::-1]:
            self.backend.bprop_fc(self.weights_rec,
                                  self.deltas[tau - layer - 1].transpose(),
                                  out=self.berror)
            self.deltas[tau - layer] = self.berror * self.pre_act_list[layer]
            self.backend.update_fc(self.deltas[tau - (layer + 1)].transpose(),
                                   self.output_list[layer].transpose(),
                                   out=self.temp_rec)
            self.updates_rec += self.temp_rec
            self.backend.update_fc(self.deltas[tau - layer].transpose(),
                                   inputs[batch_inx[:, layer]].transpose(),
                                   out=self.temp_in)
            self.updates += self.temp_in

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)
        self.learning_rule.apply_rule_rec(self.weights_rec,
                                          self.updates_rec, epoch)


class LayerWithNoBiasDist(LayerWithNoBias):

    """
    MPI Distributed
    Single NNet layer with no bias node
    """

    def adjust_for_dist(self):
        # indices of the input layer in weight matrix
        in_indices = []
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
                    in_indices.extend(
                        range(current_index, current_index + self.ifmshape[1]))
                    current_index += self.global_width
        elif self.prev_layer == 'LayerWithNoBiasDist':
            in_indices = self.in_indices
        else:
            raise ValueError('Unsupported previous layer for '
                             'LayerWithNoBiasDist')

        self.weights = self.weights.take(in_indices, axis=1)

        self.updates = self.backend.empty(self.weights.shape)
        self.learning_rule.allocate_state(self.updates)
        self.delta_ = self.backend.empty((self.nout_, self.batch_size))
        self.delta_gather = self.backend.empty(
            (self.nout, self.batch_size * MPI.COMM_WORLD.size))
        if self.pos > 0:
            # This is storage for the backward propagated error.
            self.berror = self.backend.empty((self.nin, self.batch_size))

    def fprop(self, inputs):
        # dot product is distributed across nodes
        self.backend.fprop_fc(inputs, self.weights, out=self.pre_act)
        # accumulate the pre_act values before applying non-linearity
        self.pre_act._tensor = MPI.COMM_WORLD.reduce(
            self.pre_act.raw(), op=MPI.SUM, root=0)
        # apply non-linearity on the output node
        if MPI.COMM_WORLD.rank == 0 and self.activation is not None:
            # this stores the derivatives in self.pre_act
            self.activation.apply_both(self.backend, self.pre_act, self.output)
        # strictly, following line not needed for top-most layer
        self.output._tensor = MPI.COMM_WORLD.bcast(self.output.raw())
        # broadcast back the pre_act values for bprop.
        # note: suboptimal for dist implementation,
        # but a consequence of reusing the pre_act buffer for fprop and bprop
        self.pre_act._tensor = MPI.COMM_WORLD.bcast(self.pre_act.raw())

    def bprop(self, error, inputs):
        # comment if not using denominator term in cross_entropy
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act_, out=error)
        if self.nout_ != self.nout:
            MPI.COMM_WORLD.Allgather(
                error.raw(), self.delta_gather._tensor)
            # todo: only supported in numpy backend for now
            self.delta_._tensor = np.vstack(
                np.split(self.delta_gather.raw(), MPI.COMM_WORLD.size, axis=1))
            if self.pos > 0:
                self.backend.bprop_fc(self.delta_, self.weights,
                                      out=self.berror)
            self.backend.update_fc(self.delta_, inputs, out=self.updates)
        else:
            if self.pos > 0:
                self.backend.bprop_fc(error, self.weights,
                                      out=self.berror)
            self.backend.update_fc(error, inputs, out=self.updates)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class BranchLayer(YAMLable):

    """
    Branch layer is composed of a list of other layers
    during fprop, it concatenates the component outputs and passes it on
    during bprop, it splits the backward errors into the components and
        accumulates into a common berror
    """

    def __init__(self, name, backend, batch_size, pos, nin, sublayers,
                 output_dtype=None, berror_dtype=None):
        self.name = name
        self.backend = backend
        self.nin = nin
        self.nout = 0
        self.sublayers = sublayers
        self.nsublayers = len(self.sublayers)
        self.startidx = [0]*len(self.sublayers)
        self.endidx = [0]*len(self.sublayers)

        for i in range(self.nsublayers):
            self.nout += self.sublayers[i].nout
            self.endidx[i] = self.nout
            if i > 0:
                self.startidx[i] = (self.startidx[i-1] +
                                    self.sublayers[i-1].nout)

        self.output = backend.empty((self.nout, batch_size), output_dtype)
        self.pos = pos
        if pos > 0:
            self.berror = backend.empty((nin, batch_size), berror_dtype)

    def fprop(self, inputs):
        for (sublayer, s_idx, e_idx) in zip(self.sublayers,
                                            self.startidx, self.endidx):
            sublayer.fprop(inputs)
            self.output[s_idx:e_idx] = sublayer.output

    def bprop(self, error, inputs):
        for (sublayer, s_idx, e_idx) in zip(self.sublayers,
                                            self.startidx, self.endidx):
            sublayer.bprop(error[s_idx:e_idx], inputs)

        if self.pos > 0:
            self.berror[:] = self.backend.wrap(0.0)
            for sublayer in self.sublayers:
                self.backend.add(self.berror, sublayer.berror, out=self.berror)

    def update(self, epoch):
        for sublayer in self.sublayers:
            sublayer.update(epoch)

    def set_train_mode(self, mode):
        for sublayer in self.sublayers:
            sublayer.set_train_mode(mode)


class DropOutLayer(YAMLable):

    """
    Dropout layer randomly kills activations from being passed on at each
    fprop call.
    Uses parameter 'keep' as the threshhold above which to retain activation.
    During training, the mask is applied, but during inference, we switch
    off the random dropping.
    Make sure to set train mode to False during inference.
    """

    def __init__(self, name, backend, batch_size, pos, nin, keep,
                 output_dtype=None, berror_dtype=None):
        self.name = name
        self.backend = backend
        self.activation = None
        self.nin = nin
        self.nout = nin
        self.keep = keep
        self.keepmask = backend.empty((nin, batch_size))
        self.train_mode = True
        self.output = self.backend.empty((self.nout, batch_size), output_dtype)
        self.pos = pos
        if pos > 0:
            self.berror = backend.empty((nin, batch_size), berror_dtype)

    def fprop(self, inputs):
        if (self.train_mode):
            self.backend.fill_uniform_thresh(self.keepmask, self.keep)
            self.backend.multiply(self.keepmask, inputs, out=self.output)
        else:
            self.backend.multiply(self.backend.wrap(self.keep), inputs,
                                  out=self.output)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.multiply(error, self.keepmask, out=self.berror)

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        self.train_mode = False


class DataLayer(YAMLable):

    """
    Data Layer takes datasets as input and on fprop, passes forward the latest
    batch
    """

    def __init__(self, name, backend, batch_size, datasets):
        self.name = name
        self.backend = backend
        self.batch_size = batch_size
        self.datasets = datasets
        self.output = self.backend.empty((self.nout, batch_size))
        self.current_batch = 0
        self.partition = None

    def init_datasets(self, train=True, test=True, validation=True):
        self.train = train
        self.test = test
        self.validation = validation
        self.inputs = self.datasets[0].get_inputs(
            train=self.train, test=self.test, validation=self.validation)
        self.targets = self.datasets[0].get_targets(
            train=self.train, test=self.test, validation=self.validation)

    def select_partition(self, partition):
        self.current_batch = 0
        self.partition = partition

    def fprop(self, dummyvar):
        if not self.partition:
            raise ValueError('Dataset partition must be selected prior to use')

        if self.partition not in self.inputs:
            raise ValueError('Partition does not exist')

        raise NotImplementedError('Have to implement DataLayer')

    def bprop(self, error, inputs):
        pass

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        pass


class RBMLayer(Layer):

    """
    CD1 training layer for RBM
    """

    def __init__(self, name, backend, batch_size, pos, nin,
                 nout, activation, weight_init, learning_rule):
        super(RBMLayer, self).__init__(name, backend, batch_size, pos,
                                       nin, nout, weight_init,
                                       learning_rule, activation)
        self.p_hid_plus = backend.empty((self.nout, batch_size))
        self.s_hid_plus = backend.empty((self.nout, batch_size))
        self.p_hid_minus = backend.empty((self.nout, batch_size))
        self.p_plus = backend.empty((self.nout, nin))
        self.p_minus = backend.empty((self.nout, nin))
        self.diff = backend.empty((self.nout, nin))
        self.learning_rule = learning_rule
        self.learning_rule.allocate_state(self.diff)
        self.neg_pre_act = backend.empty((self.nin, batch_size))
        self.x_minus = backend.empty((self.nin, batch_size))
        self.output = backend.empty((self.nin-1, batch_size))

    def positive(self, inputs):
        """
        Positive / upward pass of the CD1 RBM

        Arguments:
           inputs (neon.datasets.dataset.Dataset): dataset upon which
                                                      to operate
        """
        inputs = self.backend.append_bias(inputs)
        self.backend.fprop_fc(inputs, self.weights, out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_plus)
        self.backend.update_fc(self.p_hid_plus, inputs, out=self.p_plus)
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
        self.backend.bprop_fc(self.s_hid_plus, self.weights,
                              out=self.neg_pre_act)
        self.activation.apply_function(self.backend, self.neg_pre_act,
                                       self.x_minus)
        self.backend.fprop_fc(self.x_minus, self.weights, out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_minus)
        self.output[:] = self.x_minus[:-1]

    def update(self, epoch):
        """
        CD1 weight update

        Arguments:
            epoch: not used, for future compatibility
        """
        self.backend.update_fc(self.p_hid_minus, self.x_minus,
                               out=self.p_minus)
        self.backend.subtract(self.p_plus, self.p_minus, out=self.diff)
        self.learning_rule.apply_rule(self.weights, self.diff, epoch)


class LocalLayer(YAMLable):

    """
    Base class for locally connected layers.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, pooling=False,
                 activation=None, pad=0):
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

        self.ofmheight = np.int(
            np.ceil((self.ifmheight - self.fheight + 2. * pad) / stride)) + 1
        self.ofmwidth = np.int(
            np.ceil((self.ifmwidth - self.fwidth + 2. * pad) / stride)) + 1
        self.pad = -pad
        self.ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nin = nifm * self.ifmsize

        if pos > 0:
            self.berror = backend.empty((self.nin, batch_size))
            self.berrorbuf = backend.empty((self.ifmsize, batch_size * nifm))

        self.fsize = nifm * self.fheight * self.fwidth
        ofmstarts = backend.array(range(0, (self.ofmsize * nofm),
                                        self.ofmsize)).raw()
        self.ofmlocs = backend.empty((self.ofmsize, nofm), dtype='i32')
        for dst in range(self.ofmsize):
            self.ofmlocs[dst] = backend.wrap(ofmstarts + dst)

        # Figure out the connections with the previous layer.
        if pooling is True:
            self.links = backend.empty(
                (self.ofmsize, fshape[0] * fshape[1]), dtype='i32')
            self.outputbuf = backend.empty((self.ofmsize, batch_size * nifm))
            if pos > 0:
                self.berrorbuf = backend.empty((self.ifmsize,
                                                batch_size * nifm))
        else:
            self.links = backend.empty(
                (self.ofmsize, self.fsize), dtype='i32')
        # This variable tracks the top left corner of the receptive field.
        src = 0
        for dst in range(self.ofmsize):
            # Collect the column indices for the
            # entire receptive field.
            colinds = []
            for row in range(self.fheight):
                start = src + row * self.ifmwidth
                colinds += range(start, start + self.fwidth)
            fminds = colinds[:]
            if pooling is False:
                for ifm in range(1, nifm):
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
        norms = self.backend.norm(weights, order=2, axis=1)
        self.backend.divide(weights,
                            norms.reshape((norms.shape[0], 1)),
                            out=weights)

    def fprop(self, inputs):
        raise NotImplementedError('This class should not be instantiated.')

    def set_train_mode(self, mode):
        pass


class LocalLayerDist(LocalLayer):

    """
    Base class for locally connected layers.
    """
    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, pooling=False,
                 activation=None, pad=0):
        self.name = name
        self.backend = backend
        self.activation = activation
        self.pad = pad
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape
        self.fshape = fshape
        self.fheight, self.fwidth = fshape
        self.batch_size = batch_size
        self.pos = pos
        self.learning_rule = learning_rule
        self.ofmheight = (self.ifmheight - self.fheight) / stride + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / stride + 1
        self.ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.nin = nifm * self.ifmsize
        # if pos > 0:
        #    self.berror = backend.empty((batch_size, self.nin), dtype=dtype)
        self.nifm = nifm
        self.nofm = nofm
        self.fsize = nifm * self.fheight * self.fwidth
        self.stride = stride
        self.pooling = pooling

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
            self.berror = self.backend.empty((self.nin, self.batch_size))

        ofmstarts = self.backend.array(range(0, (self.ofmsize * self.nofm),
                                             self.ofmsize))

        self.ofmlocs = self.backend.empty((self.ofmsize, self.nofm),
                                          dtype='i32')
        for dst in range(self.ofmsize):
            self.ofmlocs[dst, :] = ofmstarts + dst
        # stores the flattened px location across
        # ofm in columns

        # Figure out the connections with the previous layer.
        if self.pooling is True:
            self.links = self.backend.empty(
                (self.ofmsize, self.fshape[0] * self.fshape[1]), dtype='i32')
            self.outputbuf = self.backend.empty((self.ofmsize,
                                                 self.batch_size * self.nifm))
            if self.pos > 0:
                self.berrorbuf = self.backend.empty(
                    (self.ifmsize, self.batch_size * self.nifm))
        else:
            self.links = self.backend.empty(
                (self.ofmsize, self.fsize), dtype='i32')
        # This variable tracks the top left corner of the receptive field.
        src = 0
        for dst in range(self.ofmsize):
            # Collect the column indices for the
            # entire receptive field.
            colinds = []
            for row in range(self.fheight):
                start = src + row * self.ifmwidth
                colinds += range(start, start + self.fwidth)
            fminds = colinds[:]
            if self.pooling is False:
                for ifm in range(1, self.nifm):
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
        self.output = self.backend.empty((self.nout, self.batch_size))


class ConvLayer(LocalLayer):

    """
    Convolutional layer.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, weight_init, activation=None,
                 pad=0):
        if pad != 0 and isinstance(backend, CPU):
            raise NotImplementedError('pad != 0, for CPU backend in ConvLayer')
        super(ConvLayer, self).__init__(name, backend, batch_size, pos,
                                        learning_rule, nifm, nofm,
                                        ifmshape, fshape, stride,
                                        activation=activation,
                                        pad=pad)
        self.nout = self.ofmsize * nofm
        self.weights = backend.gen_weights((self.fsize, nofm),
                                           weight_init)
        self.output = backend.empty((self.nout, batch_size))
        self.updates = backend.empty(self.weights.shape)
        self.prodbuf = backend.empty((nofm, batch_size))
        self.bpropbuf = backend.empty((self.fsize, batch_size))
        self.updatebuf = backend.empty(self.weights.shape)
        self.learning_rule.allocate_state(self.updates)
        if activation is not None:
            self.pre_act = backend.empty((self.nout, batch_size))
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
                                self.ofmlocs, self.pad, self.stride, self.nifm,
                                1, self.prodbuf)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs):
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)
        if self.pos > 0:
            self.backend.bprop_conv(self.weights, error, self.berror,
                                    self.links, self.ifmshape, self.ofmshape,
                                    self.ofmlocs, self.pad, self.stride,
                                    self.nifm, 1, self.bpropbuf)
        self.backend.update_conv(self.weights, inputs, error, self.updates,
                                 self.links, self.ifmshape, self.ofmshape,
                                 self.ofmlocs, self.pad, self.stride,
                                 self.nifm, 1, self.fwidth, self.updatebuf)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)


class ConvLayerDist(LocalLayerDist, ConvLayer):

    """
    Distributed convolutional layer.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, weight_init, activation=None,
                 pad=0):
        if pad != 0:
            raise NotImplementedError('Pad != 0, for ConvLayerDist')
        super(ConvLayerDist, self).__init__(name, backend, batch_size, pos,
                                            learning_rule, nifm, nofm,
                                            ifmshape, fshape, stride,
                                            activation=activation, pad=pad)
        self.nout = self.ofmsize * nofm
        self.weights = backend.gen_weights((self.fsize, nofm),
                                           weight_init)
        self.output = backend.empty((self.nout, batch_size))
        self.updates = backend.empty(self.weights.shape)
        self.prodbuf = backend.empty((nofm, batch_size))
        self.bpropbuf = backend.empty((self.fsize, batch_size))
        self.updatebuf = backend.empty((self.fsize, nofm))
        self.learning_rule.allocate_state(self.updates)
        if activation is not None:
            self.pre_act = backend.empty((self.nout, batch_size))
            raise NotImplementedError('TODO')
        else:
            self.pre_act = self.output

    def adjust_for_dist(self):
        self.ifmshape = self.input.local_array.ifmshape
        super(ConvLayerDist, self).adjust_for_dist(self.ifmshape)
        self.nout = self.ofmsize * self.nofm
        self.output = self.backend.empty((self.nout, self.batch_size))
        if self.activation is not None:
            self.pre_act = self.backend.empty((self.nout, self.batch_size))
            raise NotImplementedError('TODO')
        else:
            self.pre_act = self.output

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(ConvLayerDist, self).fprop(inputs)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_conv(self.weights, error, self.berror,
                                    self.links, self.ifmshape, self.ofmshape,
                                    self.ofmlocs, 0, self.stride, self.nifm,
                                    1, self.bpropbuf)
        # accumulate updates across tiles for all filters
        # if want to keep weights unshared across nodes, could not do the
        # transfers here
        self.updates._tensor = MPI.COMM_WORLD.reduce(
            self.updates.raw(), op=MPI.SUM, root=0)
        self.updates._tensor = MPI.COMM_WORLD.bcast(self.updates.raw())
        self.backend.update_conv(self.weights, inputs, error, self.updates,
                                 self.links, self.ifmshape, self.ofmshape,
                                 self.ofmlocs, 0, self.stride, self.nifm,
                                 1, self.fwidth, self.updatebuf)

    def update(self, epoch):
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
        self.output = backend.empty((self.nout, batch_size))
        self.weights = self.backend.gen_weights((self.nout, self.fsize),
                                                weight_init)

        self.normalize_weights(self.weights)
        self.updates = backend.empty(self.weights.shape)
        self.prodbuf = backend.empty((nofm, batch_size))
        self.bpropbuf = backend.empty((self.fsize, batch_size))
        self.updatebuf = backend.empty((nofm, self.fsize))
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
        cost.set_outputbuf(self.defilter.output)
        error = cost.apply_derivative(inputs)
        self.backend.divide(error, self.backend.wrap(inputs.shape[1]),
                            out=error)
        self.defilter.bprop(error, self.output)
        self.defilter.update(epoch)
        # Now backward propagate the gradient of the output of the
        # pooling layer.
        error = ((self.sparsity / inputs.shape[1]) *
                 (self.backend.ones(self.pooling.output.shape)))
        self.pooling.bprop(error, self.output)
        # Aggregate the errors from both layers before back propagating
        # through the current layer.
        berror = self.defilter.berror + self.pooling.berror
        self.bprop(berror, inputs)
        self.update(epoch)
        rcost = cost.apply_function(inputs)
        spcost = self.sparsity * self.pooling.output.sum()
        return rcost, spcost

    def fprop(self, inputs):
        for dst in range(self.ofmsize):
            rflinks = self.rlinks[dst]
            # We use a different filter for each receptive field.
            # size-guide
            # inputs.take: mbs x (ifmsize*nifm) ->  mbs x (fmsize*nifm)
            # self.weights: (nout x (ifmsize*nifm)).T -> (fsize x nofm)
            self.backend.dot(self.weights.take(self.ofmlocs[dst],
                                               axis=0),
                             inputs.take(rflinks, axis=0),
                             out=self.prodbuf)
            # size: # mbs x nofm
            self.output[self.ofmlocs[dst], :] = self.prodbuf

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.clear(self.berror)
            for dst in range(self.ofmsize):
                # Use the same filter that was used for forward propagation
                # of this receptive field.
                # size-guide
                # self.delta.take: # mbs x nofm
                # self.weights.take: # (nofm x fsize )
                self.backend.dot(
                    self.weights.take(self.ofmlocs[dst], axis=0).transpose(),
                    error.take(self.ofmlocs[dst], axis=0), self.bpropbuf)
                rflinks = self.rlinks[dst]
                self.backend.add(self.bpropbuf,
                                 self.berror.take(rflinks, axis=0),
                                 out=self.bpropbuf)
                self.berror[rflinks, :] = self.bpropbuf

        for dst in range(self.ofmsize):
            rflinks = self.rlinks[dst]
            delta_slice = error.take(self.ofmlocs[dst], axis=0)
            self.backend.dot(delta_slice,
                             inputs.take(rflinks, axis=0).transpose(),
                             out=self.updatebuf)
            self.updates[self.ofmlocs[dst]] = self.updatebuf

    def update(self, epoch):
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

        self.output = self.backend.empty((self.nout, self.batch_size))

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
        self.updates = self.backend.empty(self.weights.shape)
        self.learning_rule.allocate_state(self.updates)
        self.prodbuf = self.backend.empty((self.nofm, self.batch_size))
        self.bpropbuf = self.backend.empty((self.fsize, self.batch_size))
        self.updatebuf = self.backend.empty((self.nofm, self.fsize))

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
        self.defilter.temp1 = [self.backend.empty(
            (self.input.local_array.local_array_size, self.batch_size))]
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
        self.backend.divide(error, self.backend.wrap(inputs.shape[1]),
                            out=error)
        self.defilter.bprop(error, self.output)
        self.defilter.update(epoch)
        # Now backward propagate the gradient of the output of the
        # pooling layer.
        error = ((self.sparsity / inputs.shape[1]) *
                 (self.backend.ones(self.pooling.output.shape)))
        self.pooling.bprop(error, self.output)
        berror = self.defilter.berror + (
            self.pooling.input.get_bprop_view(self.pooling.berror))
        self.bprop(berror, inputs)
        self.update(epoch)
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
        self.output = prev.backend.empty((prev.nin, prev.batch_size))
        if tied_weights is True:
            # Share the weights with the previous layer.
            self.weights = prev.weights
        else:
            self.weights = prev.weights.copy()
        self.updates = prev.backend.empty(self.weights.shape)
        self.prodbuf = prev.backend.empty((prev.fsize, prev.batch_size))
        self.bpropbuf = prev.backend.empty((prev.nofm, prev.batch_size))
        self.updatebuf = prev.backend.empty((prev.nofm, prev.fsize))
        self.berror = prev.backend.empty((prev.nout, prev.batch_size))
        self.temp = [prev.backend.empty(self.output.shape)]
        self.learning_rule = prev.learning_rule
        self.learning_rule.set_pretrain_mode(True)
        self.backend = prev.backend
        self.rlinks = prev.rlinks
        self.prev = prev

    def fprop(self, inputs):
        self.backend.clear(self.output)
        for dst in range(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            # size guide:
            # inputs[:, self.prev.ofmlocs[dst]]: mbs x nout -> mbs x nofm
            # self.weights.take: nofm x ifmsize
            self.backend.dot(self.weights.take(self.prev.ofmlocs[dst],
                                               axis=0).transpose(),
                             inputs[self.prev.ofmlocs[dst], :],
                             out=self.prodbuf)
            self.output[rflinks, :] += self.prodbuf

    def bprop(self, error, inputs):
        for dst in range(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(self.weights.take(self.prev.ofmlocs[dst],
                                               axis=0),
                             error[rflinks, :],
                             out=self.bpropbuf)
            self.berror[self.prev.ofmlocs[dst], :] = self.bpropbuf
            delta_slice = error[rflinks, :]
            self.backend.dot(inputs[self.prev.ofmlocs[dst], :],
                             delta_slice.transpose(),
                             out=self.updatebuf)
            self.updates[self.prev.ofmlocs[dst]] = self.updatebuf

    def update(self, epoch):
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
        self.maxinds = backend.empty((self.ofmsize, batch_size * nifm),
                                     dtype='i16')
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.empty((self.nout, batch_size))
        assert fshape[0] * fshape[1] <= 2 ** 15

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
            inputs, self.output, self.outputbuf, self.links,
            self.ifmshape, self.ofmshape, self.fshape, 0,
            self.stride, self.nifm, self.maxinds)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_mpool(
                inputs, self.output,
                error, self.berror, self.berrorbuf, self.links,
                self.ifmshape, self.ofmshape, self.fshape, 0, self.stride,
                self.nifm, self.maxinds)

    def update(self, epoch):
        pass


class MaxPoolingLayerDist(LocalLayerDist, MaxPoolingLayer):

    """
    Distributed Max pooling layer.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(MaxPoolingLayerDist, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm, ifmshape,
            fshape, stride, pooling=True)
        self.maxinds = backend.empty((self.ofmsize, batch_size * nifm),
                                     dtype='i16')
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.empty((self.nout, batch_size))

    def adjust_for_dist(self):
        self.ifmshape = self.input.local_array.ifmshape
        super(MaxPoolingLayerDist, self).adjust_for_dist(self.ifmshape)
        self.prodbuf = self.backend.empty(
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
        self.prodbuf = self.backend.empty((self.fshape[0] * self.fshape[1],
                                           batch_size * nifm))
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.empty((self.nout, batch_size))

    def __str__(self):
        return ("L2PoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        self.backend.fprop_l2pool(
            inputs, self.output, self.outputbuf, self.links,
            self.ifmshape, self.ofmshape, self.fshape,
            0, self.stride, self.nifm)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_l2pool(
                inputs, self.output, error, self.berror, self.berrorbuf,
                self.links, self.ifmshape, self.ofmshape, self.fshape,
                0, self.stride, self.nifm, self.prodbuf)

    def update(self, epoch):
        pass


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
        self.prodbuf = self.backend.empty((self.fshape[0] * self.fshape[1],
                                           batch_size * nifm))
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.empty((self.nout, batch_size))

    def adjust_for_dist(self):
        # shape with halos
        ifmshape = self.input.local_array.ifmshape
        super(L2PoolingLayerDist, self).adjust_for_dist(ifmshape)
        self.prodbuf = self.backend.empty(
            (self.fshape[0] * self.fshape[1], self.batch_size * self.nifm))

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(L2PoolingLayerDist, self).fprop(inputs)

    def bprop(self, error, inputs_):
        # redo-ing get_fprop_view, could cache for speed-up
        inputs = self.input.get_fprop_view(inputs_)
        super(L2PoolingLayerDist, self).bprop(error, inputs)


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
        self.output = self.backend.empty((self.nout, batch_size))

    def __str__(self):
        return ("AveragePoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        self.backend.fprop_apool(
            inputs, self.output, self.outputbuf, self.links,
            self.ifmshape, self.ofmshape, self.fshape,
            0, self.stride, self.nifm)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_apool(
                self.output, error, self.berror, self.berrorbuf, self.links,
                self.ifmshape, self.ofmshape, self.fshape,
                0, self.stride, self.nifm)

    def update(self, epoch):
        pass


class AveragePoolingLayerDist(LocalLayerDist, AveragePoolingLayer):

    """
    Distributed Average pooling layer.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride):
        super(AveragePoolingLayerDist, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True)
        self.prodbuf = self.backend.empty((batch_size * nifm,
                                           self.fshape[0] * self.fshape[1]))
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.empty((self.nout, batch_size))

    def adjust_for_dist(self):
        # shape with halos
        ifmshape = self.input.local_array.ifmshape
        super(AveragePoolingLayerDist, self).adjust_for_dist(ifmshape)
        self.prodbuf = self.backend.empty(
            (self.batch_size * self.nifm, self.fshape[0] * self.fshape[1]))

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(AveragePoolingLayerDist, self).fprop(inputs)

    def bprop(self, error, inputs_):
        # redo-ing get_fprop_view, could cache for speed-up
        inputs = self.input.get_fprop_view(inputs_)
        super(AveragePoolingLayerDist, self).bprop(error, inputs)


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
        self.output = backend.empty((self.nout, batch_size))
        self.prodbuf = backend.empty((nofm, batch_size))

    def fprop(self, inputs):
        for dst in range(self.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(self.weights, inputs.take(rflinks, axis=0),
                             out=self.prodbuf)
            self.output[self.ofmlocs[dst], :] = self.prodbuf


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

        self.exinputs = self.backend.empty((nifm * self.exifmsize, batch_size))
        self.rexinputs = self.exinputs.reshape((self.nifm,
                                                self.exifmheight,
                                                self.exifmwidth,
                                                batch_size))
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
        self.rmeanfm = self.meanfm.reshape((1,
                                            self.ifmheight,
                                            self.ifmwidth,
                                            batch_size))

        self.output = backend.empty((self.nout, batch_size))
        self.routput = self.output.reshape((nifm,
                                            self.ifmheight,
                                            self.ifmwidth,
                                            batch_size))
        self.subout = backend.empty(self.output.shape)
        self.rsubout = self.subout.reshape(self.routput.shape)
        self.subtemp = backend.empty(self.output.shape)
        self.rsubtemp = self.subtemp.reshape(self.routput.shape)
        if pos > 0:
            self.diverror = backend.empty((self.nin, batch_size))
            self.exerror = self.backend.empty((nifm * self.exifmsize,
                                               batch_size))
            self.rexerror = self.exerror.reshape((nifm,
                                                  self.exifmheight,
                                                  self.exifmwidth,
                                                  batch_size))
            self.prodbuf = self.backend.empty((self.fsize, batch_size))
            self.bprop_filters = self.backend.empty((nifm,
                                                     self.filters.shape[0],
                                                     self.filters.shape[1]))
            self.sqtemp = backend.empty(self.output.shape)
            for fm in range(nifm):
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
        filters = self.backend.empty((count, shape[0], shape[1]))
        filters[:] = single

        filters = filters.reshape((1, count * shape[0] * shape[1]))
        return filters

    def copy_to_inset(self, canvas, inset, start_row, start_col):
        canvas[:, start_row:(canvas.shape[1] - start_row),
               start_col:(canvas.shape[2] - start_col)] = inset

    def copy_from_inset(self, canvas, start_row, start_col):
        return canvas[:, self.start_row:(canvas.shape[1] - start_row),
                      self.start_col:(canvas.shape[2] - start_col)]

    def fprop_sub_normalize(self, inputs):
        rinputs = inputs.reshape((self.nifm, self.ifmheight, self.ifmwidth,
                                  self.batch_size))
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
        self.berror = self.berror.reshape((self.nin, self.batch_size))

    def bprop_sub_normalize(self, error, inputs):
        self.backend.clear(self.exerror)
        for fm in range(self.nifm):
            for dst in range(self.conv.ofmsize):
                rflinks = self.conv.rlinks[dst]
                loc = self.conv.ofmlocs[dst].raw() + self.conv.ofmsize * fm
                filt = self.bprop_filters[fm]
                self.backend.multiply(error[loc], filt.transpose(),
                                      out=self.prodbuf)
                self.exerror[rflinks] -= self.prodbuf
        self.reshape_error()

    def bprop_div_normalize(self, error, inputs):
        self.backend.clear(self.exerror)
        self.backend.cube(self.output, out=self.diverror)
        self.subtemp[:] = self.subout
        assert self.diverror[self.subout.raw() == 0].sum() == 0.0
        self.subout[self.subout.raw() == 0] = 1.0
        self.backend.square(self.subout, out=self.sqtemp)
        # this is for the non-padded, non-halo matrix only
        self.backend.divide(self.diverror, self.sqtemp, out=self.diverror)

        for fm in range(self.nifm):
            for dst in range(self.conv.ofmsize):
                # self.conv.ofmlocs is over 1 fm only
                loc = self.conv.ofmlocs[dst].raw() + self.conv.ofmsize * fm
                divout = self.output.take(loc, axis=0)
                subout = self.subout.take(loc, axis=0)
                assert divout[subout.raw() == 0].sum() == 0
                subout[subout.raw() == 0.0] = 1.0
                self.backend.divide(divout, subout, out=divout)

                rflinks = self.conv.rlinks[dst]
                self.copy_to_inset(self.rexinputs, self.rsubtemp,
                                   self.start_row, self.start_col)
                rrexinputs = self.rexinputs.reshape(
                    (self.nifm * self.exifmsize, self.batch_size))
                frame = rrexinputs.take(rflinks, axis=0)
                self.backend.multiply(frame, self.filters.transpose(),
                                      out=frame)
                self.backend.multiply(frame, self.diverror[loc], out=frame)
                rframe = frame.reshape((self.nifm, self.fheight, self.fwidth,
                                        self.batch_size))
                # this is working on the g2/y2 term
                rframe[fm:(fm + 1),
                       self.fheight / 2, self.fwidth / 2] -= divout
                self.backend.multiply(error[loc],
                                      frame, out=frame)
                self.exerror[rflinks] -= frame
        self.reshape_error()

    def bprop(self, error, inputs):
        if self.pos > 0:
            # note: have to account for halos + padding after each step
            self.bprop_div_normalize(error, inputs)
            self.bprop_sub_normalize(self.berror, inputs)

    def bprop_fast(self, error, inputs):
        """
        An incorrect, but much faster version of backprop.
        """
        if self.pos > 0:
            self.berror[:] = error

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        pass


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

        self.exinputs = self.backend.empty((self.nifm * self.exifmsize,
                                            self.batch_size,))
        self.rexinputs = self.exinputs.reshape((self.nifm,
                                                self.exifmheight,
                                                self.exifmwidth,
                                                self.batch_size))
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
        self.rmeanfm = self.meanfm.reshape((1, output_height, output_width,
                                            self.batch_size, ))

        self.output = self.backend.empty((self.nout, self.batch_size))
        self.routput = self.output.reshape((self.nifm,
                                            output_height, output_width,
                                            self.batch_size))

        self.temp1 = self.backend.empty(self.output.shape)
        self.rtemp1 = self.temp1.reshape(self.routput.shape)
        self.temp2 = self.backend.empty(self.output.shape)
        self.rtemp2 = self.temp2.reshape(self.routput.shape)
        self.subout = self.backend.empty(self.output.shape)
        self.rsubout = self.subout.reshape(self.routput.shape)
        self.subtemp = self.backend.empty(self.output.shape)
        self.rsubtemp = self.subtemp.reshape(self.routput.shape)
        self.subtemp2 = self.backend.empty((self.nin, self.batch_size))
        self.rsubtemp2 = self.subtemp2.reshape((self.nifm,
                                                self.ifmheight, self.ifmwidth,
                                                self.batch_size))

        if self.pos > 0:
            # changed to nout for bprop in dist version, compared to nin in
            # non-dist version
            self.diverror = self.backend.empty(
                (self.nout, self.batch_size))
            self.exerror = self.backend.empty((self.nifm * self.exifmsize,
                                               self.batch_size))
            self.rexerror = self.exerror.reshape((self.nifm,
                                                  self.exifmheight,
                                                  self.exifmwidth,
                                                  self.batch_size))
            self.prodbuf = self.backend.empty(
                (self.fsize, self.batch_size))
            self.bprop_filters = self.backend.empty((self.nifm,
                                                     self.filters.shape[0],
                                                     self.filters.shape[1]))
            self.sqtemp = self.backend.empty(self.output.shape)
            for fm in range(self.nifm):
                self.bprop_filters[fm] = self.filters.copy()
                rfilter = self.bprop_filters[fm].reshape(
                    (self.nifm, self.fheight, self.fwidth))
                rfilter[fm, self.fheight / 2, self.fwidth / 2] -= 1.0

    def copy_to_inset(self, canvas, inset, start_row, start_col):
        canvas[:, start_row:start_row + inset.shape[1],
               start_col:start_col + inset.shape[2], :] = inset

    def copy_from_inset(self, canvas, start_row, start_col):
        return canvas[:, start_row:start_row + self.ifmheight,
                      start_col:start_col + self.ifmwidth, :]

    def fprop_sub_normalize(self, inputs):
        rinputs = inputs.reshape((self.nifm,
                                  self.ifmheight, self.ifmwidth,
                                  self.batch_size))
        self.copy_to_inset(self.rexinputs, rinputs,
                           self.start_row, self.start_col)
        # Convolve with gaussian filters to obtain a "mean" feature map.
        self.conv.fprop(self.exinputs)
        # rinputs includes halos but not padding
        self.backend.subtract(
            self.rexinputs[:,
                           self.start_row2:(
                               self.rexinputs.shape[1] - self.start_row2),
                           self.start_col2:(
                               self.rexinputs.shape[2] - self.start_col2), :],
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

    def bprop_div_normalize(self, error, inputs):
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
            for dst in range(self.conv.ofmsize):
                # self.conv.ofmlocs is over 1 fm only
                loc = self.conv.ofmlocs[dst].raw() + self.conv.ofmsize * fm
                divout = self.output.take(loc, axis=0)
                subout = self.subout.take(loc, axis=0)
                assert divout[subout.raw() == 0].sum() == 0
                subout[subout.raw() == 0.0] = 1.0
                self.backend.divide(divout, subout, out=divout)

                rflinks = self.conv.rlinks[dst]
                self.copy_to_inset(self.rexinputs, self.rsubtemp2,
                                   self.start_row, self.start_col)
                rrexinputs = self.rexinputs.reshape(
                    (self.nifm * self.exifmsize, self.batch_size))
                frame = rrexinputs.take(rflinks, axis=0)
                self.backend.multiply(frame, self.filters.transpose(),
                                      out=frame)
                self.backend.multiply(frame, self.diverror[loc, :], out=frame)
                rframe = frame.reshape((self.nifm,
                                        self.fheight, self.fwidth,
                                        self.batch_size))
                # this is working on the g2/y2 term
                rframe[fm:(fm + 1),
                       self.fheight / 2, self.fwidth / 2, :] -= divout
                self.backend.multiply(error[loc, :].repeat(self.fsize, axis=0),
                                      frame, out=frame)
                self.exerror[rflinks, :] -= frame
        self.reshape_error()

    def bprop(self, error, inputs):
        if self.pos > 0:
            # note: have to account for halos + padding after each step
            self.bprop_div_normalize(error, inputs)

            self.bprop_sub_normalize(self.input.get_bprop_view(self.berror),
                                     inputs)

            self.berror = (self.input.get_bprop_view(self.berror))


class CrossMapPoolingLayer(YAMLable):

    """
    Pool input feature maps by computing a weighted sum of
    corresponding spatial locations across maps. This is
    equivalent to a 1x1 convolution.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule,
                 nifm, nofm, ifmshape, weight_init, activation=None):
        self.name = name
        self.backend = backend
        self.batch_size = batch_size
        self.pos = pos
        self.learning_rule = learning_rule
        self.nifm = nifm
        self.nofm = nofm
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape
        self.activation = activation

        self.ofmshape = self.ifmshape
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ifmsize
        self.nin = nifm * self.ifmsize
        self.nout = nofm * self.ifmsize
        if pos > 0:
            self.berror = backend.empty((self.nin, batch_size))

        self.weights = backend.gen_weights((nifm, nofm),
                                           weight_init)
        assert (self.weights.raw() < 0).sum() == 0
        self.updates = backend.empty(self.weights.shape)
        self.output = backend.empty((self.nout, batch_size))
        self.updatebuf = backend.empty((1, 1))
        self.learning_rule.allocate_state(self.updates)
        if activation is not None:
            self.pre_act = backend.empty((self.nout, batch_size))
        else:
            self.pre_act = self.output

    def fprop(self, inputs):
        self.backend.fprop_cmpool(inputs, self.weights, self.ifmsize,
                                  out=self.pre_act)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs):
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)
        if self.pos > 0:
            self.backend.bprop_cmpool(error, self.weights, self.ifmsize,
                                      out=self.berror)
        self.backend.update_cmpool(error, inputs, self.ifmsize,
                                   self.updatebuf, out=self.updates)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)

    def set_train_mode(self, mode):
        pass


class CrossMapResponseNormLayer(YAMLable):

    """
    CrossMap response normalization.

    Calculates the normalization across feature maps at each pixel point.
    output will be same size as input

    The calculation is output(x,y,C) = input(x,y,C)/normFactor(x,y,C)

    where normFactor(x,y,C) is (1 + alpha * sum_ksize( input(x,y,k)^2 ))^beta

    ksize is the kernel size, so will run over the channel index with no
    padding at the edges of the feature map.  (so for ksize=5, at C=1, we will
    be summing the values of c=0,1,2,3)
    """

    def __init__(self, name, backend, batch_size, pos, ifmshape,
                 nifm, ksize, alpha, beta):

        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nifm = nifm
        self.nin = self.ifmsize * self.nifm
        self.nout = self.nin

        self.name = name
        self.backend = backend
        self.ifmshape = ifmshape
        self.batch_size = batch_size
        self.pos = pos

        self.ksize = ksize
        self.alpha = alpha * 1.0 / ksize
        self.beta = beta

        self.output = self.backend.empty((self.nout, self.batch_size))
        if self.pos > 0:
            self.berror = self.backend.empty((self.nin, self.batch_size))
            self.tempbuf = self.backend.empty((ifmshape[0], ifmshape[1],
                                              batch_size))

    def fprop(self, inputs):
        self.backend.fprop_cmrnorm(inputs, self.output, self.ifmshape,
                                   self.nifm, self.ksize, self.alpha,
                                   self.beta)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_cmrnorm(inputs, self.output, error, self.berror,
                                       self.ifmshape, self.nifm, self.ksize,
                                       self.alpha, self.beta, self.tempbuf)

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        pass
