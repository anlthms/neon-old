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
                 weight_init, learning_rule, prev_names=[], activation=None,
                 weight_dtype=None, updates_dtype=None, pre_act_dtype=None,
                 output_dtype=None, deltas_dtype=None):
        self.name = name
        self.backend = backend
        self.activation = activation
        self.nin = nin
        self.nout = nout
        self.weight_init = weight_init
        self.weight_dtype = weight_dtype
        self.weights = self.weight_init.generate((nout, nin), weight_dtype)
        self.weight_updates = self.backend.empty(self.weights.shape,
                                                 updates_dtype)
        self.updates_dtype = updates_dtype
        self.output = self.backend.zeros((self.nout, batch_size), output_dtype)
        self.prev_names = prev_names
        if activation is not None:
            self.pre_act = self.backend.zeros(self.output.shape,
                                              pre_act_dtype)
        else:
            self.pre_act = self.output

        self.pos = pos
        self.learning_rule = learning_rule
        self.batch_size = batch_size
        self.use_biases = 'bias_init' in weight_init.__dict__
        if self.use_biases:
            self.biases = self.backend.empty((nout, 1), weight_dtype)
            self.biases.fill(weight_init.bias_init)
            self.bias_updates = self.backend.empty(self.biases.shape,
                                                   updates_dtype)
            self.params = [self.weights, self.biases]
            self.updates = [self.weight_updates, self.bias_updates]
        else:
            self.params = [self.weights]
            self.updates = [self.weight_updates]

        self.learning_rule.allocate_state(self.updates)

        if pos > 0:
            # This is storage for the backward propagated error.
            self.deltas = self.backend.empty((nin, batch_size),
                                             deltas_dtype)
            self.deltas_dtype = deltas_dtype

    def __str__(self):
        temp = self.backend.empty((1, 1))
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
                 y_avg=float(self.backend.mean(self.output, axes=None,
                                               out=temp).asnumpyarray()),
                 y_min=float(self.backend.min(self.output, axes=None,
                                              out=temp).asnumpyarray()),
                 y_absmin=float(self.backend.min(self.backend.fabs(
                                                 self.output), axes=None,
                                                 out=temp).asnumpyarray()),
                 y_max=float(self.backend.max(self.output, axes=None,
                                              out=temp).asnumpyarray()),
                 y_dtype=self.output.dtype,
                 z_avg=float(self.backend.mean(self.pre_act, axes=None,
                                               out=temp).asnumpyarray()),
                 z_min=float(self.backend.min(self.pre_act, axes=None,
                                              out=temp).asnumpyarray()),
                 z_absmin=float(self.backend.min(self.backend.fabs(
                                                 self.pre_act), axes=None,
                                                 out=temp).asnumpyarray()),
                 z_max=float(self.backend.max(self.pre_act, axes=None,
                                              out=temp).asnumpyarray()),
                 z_dtype=self.pre_act.dtype,
                 w_avg=float(self.backend.mean(self.weights, axes=None,
                                               out=temp).asnumpyarray()),
                 w_min=float(self.backend.min(self.weights, axes=None,
                                              out=temp).asnumpyarray()),
                 w_absmin=float(self.backend.min(self.backend.fabs(
                                                 self.weights), axes=None,
                                                 out=temp).asnumpyarray()),
                 w_max=float(self.backend.max(self.weights, axes=None,
                                              out=temp).asnumpyarray()),
                 w_dtype=self.weights.dtype))

    def fprop(self, inputs):
        self.backend.fprop_fc(out=self.pre_act, inputs=inputs,
                              weights=self.weights)
        if self.use_biases is True:
            self.backend.add(self.pre_act, self.biases, out=self.pre_act)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs):
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)

        if self.pos > 0:
            self.backend.bprop_fc(out=self.deltas, weights=self.weights,
                                  deltas=error)

        self.backend.update_fc(out=self.weight_updates, inputs=inputs,
                               deltas=error)
        if self.use_biases is True:
            self.backend.sum(error, axes=1, out=self.bias_updates)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.params, self.updates, epoch)

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
                self.backend.begin()
                current_index = (cur_channel * self.global_size +
                                 self.top_left_row_output * self.global_width +
                                 self.top_left_col_output)
                for cur_row in range(self.ifmshape[0]):
                    self.backend.begin()
                    in_indices.extend(
                        range(current_index, current_index + self.ifmshape[1]))
                    current_index += self.global_width
                    self.backend.end()
                self.backend.end()
        elif self.prev_layer == 'LayerDist':
            in_indices = self.in_indices
        else:
            raise ValueError('Unsupported previous layer for '
                             'LayerDist')

        self.weights = self.weights.take(in_indices, axis=1)
        self.weight_updates = self.backend.empty(self.weights.shape)

        if self.use_biases:
            self.params = [self.weights, self.biases]
            self.updates = [self.weight_updates, self.bias_updates]
        else:
            self.params = [self.weights]
            self.updates = [self.weight_updates]

        self.learning_rule.allocate_state(self.updates)
        self.delta_ = self.backend.empty((self.nout_, self.batch_size))
        self.delta_gather = self.backend.empty(
            (MPI.COMM_WORLD.size * self.nout, self.batch_size))
        if self.pos > 0:
            # This is storage for the backward propagated error.
            self.deltas = self.backend.empty((self.nin, self.batch_size))

    def fprop(self, inputs):
        self.backend.fprop_fc(out=self.pre_act, inputs=inputs,
                              weights=self.weights)
        # accumulate the pre_act values before applying non-linearity
        self.pre_act._tensor = MPI.COMM_WORLD.reduce(
            self.pre_act.asnumpyarray(), op=MPI.SUM, root=0)
        # apply non-linearity on the output node
        if MPI.COMM_WORLD.rank == 0 and self.activation is not None:
            # this stores the derivatives in self.pre_act
            self.activation.apply_both(self.backend, self.pre_act, self.output)
        # strictly, following line not needed for top-most layer
        self.output._tensor = MPI.COMM_WORLD.bcast(self.output.asnumpyarray())
        # broadcast back the pre_act values for bprop.
        # note: suboptimal for dist implementation,
        # but a consequence of reusing the pre_act buffer for fprop and bprop
        self.pre_act._tensor = MPI.COMM_WORLD.bcast(
            self.pre_act.asnumpyarray())

    def bprop(self, error, inputs):
        """
        # numpy pseudocode for the backprop:
        # updates  = dot(error.T, inputs)        # calculate new gradient
        # weight update itself done by application of learning rule
        """
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act_, out=error)
        if self.nout_ != self.nout:
            MPI.COMM_WORLD.Allgather(
                error.asnumpyarray(), self.delta_gather._tensor)
            self.delta_._tensor = np.vstack(
                np.split(self.delta_gather.asnumpyarray(), MPI.COMM_WORLD.size,
                         axis=0))
            if self.pos > 0:
                self.backend.bprop_fc(out=self.deltas,
                                      weights=self.weights,
                                      deltas=self.delta_)
            self.backend.update_fc(out=self.weight_updates, inputs=inputs,
                                   deltas=self.delta_)
        else:
            if self.pos > 0:
                self.backend.bprop_fc(out=self.deltas,
                                      weights=self.weights,
                                      deltas=error)
            self.backend.update_fc(out=self.weight_updates, inputs=inputs,
                                   deltas=error)


class LayerMultiPass(Layer):
    """
    Single NNet layer that accumulates backpropagated error.

    Multipass indicates that multiple back propagation passes can be made
    (each corresponding to different cost), and the gradient will be
    accumulated until an update is called, at which point the gradients will
    be cleared
    """
    def __init__(self, name, backend, batch_size, pos, nin, nout,
                 weight_init, learning_rule, prev_names=[], activation=None,
                 weight_dtype=None, updates_dtype=None, pre_act_dtype=None,
                 output_dtype=None, deltas_dtype=None):
        super(LayerMultiPass, self).__init__(name, backend, batch_size,
                                             pos, nin, nout, weight_init,
                                             learning_rule,
                                             activation=activation,
                                             prev_names=prev_names)
        for uparam in self.updates:
            self.backend.begin()
            uparam[:] = 0.0
            self.backend.end()

        self.utemp = map(lambda x:
                         self.backend.empty(x.shape, self.updates_dtype),
                         self.params)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.params, self.updates, epoch)
        for uparam in self.updates:
            self.backend.begin()
            uparam[:] = 0.0
            self.backend.end()

    def bprop(self, error, inputs, useshortcut=False):
        # If we are back propagating error from more than one cost through the
        # network, and they do not cancel out nicely (softmax with mCE) then we
        # should do a full multiply against the activation derivative.
        # Otherwise just pass the error right through.

        if self.activation is not None and useshortcut is False:
            self.backend.multiply(error, self.pre_act, out=error)

        if self.pos > 0:
            self.backend.bprop_fc(out=self.deltas, weights=self.weights,
                                  deltas=error)

        self.backend.update_fc(out=self.utemp[0], inputs=inputs,
                               deltas=error)
        self.backend.add(self.utemp[0], self.weight_updates,
                         out=self.weight_updates)

        if self.use_biases is True:
            self.backend.sum(error, axes=1, out=self.utemp[1])
            self.backend.add(self.utemp[1], self.bias_updates,
                             out=self.bias_updates)


class RecurrentOutputLayer(Layer):

    """
    Derived from Layer. pre_act becomes pre_act_list, output becomes
    output_list, which are indexed by [tau], the unrolling step.
    """

    def __init__(self, name, backend, batch_size, pos, nin, nout, unrolls,
                 activation, weight_init, learning_rule, weight_dtype=None,
                 delta_dtype=None, updates_dtype=None, pre_act_dtype=None,
                 output_dtype=None, deltas_dtype=None):
        super(RecurrentOutputLayer, self).__init__(name, backend, batch_size,
                                                   pos, nin, nout, weight_init,
                                                   learning_rule,
                                                   activation=activation)
        self.pre_act_list = [self.backend.zeros((nout, batch_size),
                                                pre_act_dtype)
                             for k in range(unrolls)]
        self.output_list = [self.backend.zeros((nout, batch_size),
                                               output_dtype)
                            for k in range(unrolls)]
        self.temp_out = self.backend.zeros((nout, nin))
        # self.deltas_o = [self.backend.zeros((nout, batch_size))
        #                  for k in range(unrolls + 1)]
        if pos > 0:
            self.deltas = backend.zeros((self.nin, self.batch_size))

    def fprop(self, inputs, tau):
        self.backend.fprop_fc(self.pre_act_list[tau],
                              inputs, self.weights)
        if self.activation is not None:
            self.activation.apply_both(self.backend,
                                       self.pre_act_list[tau],
                                       self.output_list[tau])
        else:
            raise AttributeError("Urs is not cool with your missing "
                                 "activation function")

    def bprop(self, error, inputs, tau, numgrad=False):
        self.backend.multiply(error, self.pre_act_list[tau - 1], error)
        self.backend.bprop_fc(self.deltas,
                              self.weights,
                              error)
        self.backend.update_fc(out=self.temp_out,
                               inputs=inputs,
                               deltas=error)
        if numgrad is "output":
            logger.info("RecurrentOutputLayer.bprop inc out %f",
                        self.temp_out[12, 56])
        self.backend.add(self.weight_updates, self.temp_out,
                         self.weight_updates)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.params, self.updates, epoch)


class RecurrentLSTMLayer(Layer):

    """
    Hidden layer with LSTM gates.
    This is a plug in replacement for RecurrentHiddenLayer()
    """

    def __init__(self, name, backend, batch_size, pos, nin, nout, unrolls,
                 activation, gate_activation, weight_init, weight_init_rec,
                 learning_rule,
                 weight_dtype=None, delta_dtype=None, updates_dtype=None,
                 pre_act_dtype=None, output_dtype=None, deltas_dtype=None):
        """
        In this section, create buffers for the 8 weight matrices:
        two kind of inputs (x_t and h_t-1) feeding into 4 gates (input, output,
        forget, cell). In addition to weights, create buffers for preactivation
        values and for the intermediate values computed in the LSTM cell.

        """
        # super calls into Layer.__init__() for weight init.
        super(RecurrentLSTMLayer, self).__init__(name, backend, batch_size,
                                                 pos, nin, nout, weight_init,
                                                 learning_rule,
                                                 activation=activation)

        # things that are not initalized by the super class
        self.gate_activation = gate_activation  # same for activation in super
        be = backend
        net_sze = (self.nout, batch_size)  # tuple with activation size.

        # create weight matrices -- TODO: weight_init in yaml
        for a in ['i', 'f', 'o', 'g']:
            self.backend.begin()
            setattr(self, a + '_t',
                    [be.zeros(net_sze) for k in range(unrolls)])
            setattr(self, 'net_' + a,
                    [be.zeros(net_sze) for k in range(unrolls)])
            self.backend.end()

        for a in ['i', 'f', 'o', 'c']:
            self.backend.begin()
            setattr(self, 'W' + a + 'x',
                    weight_init_rec.generate((nout, nin), weight_dtype))
            setattr(self, 'W' + a + 'h', weight_init_rec.generate(
                    (nout, nout), weight_dtype))
            setattr(self, 'b_' + a, be.zeros((nout, 1)))
            setattr(self, 'W' + a + 'x_updates', be.zeros((nout, nin)))
            setattr(self, 'W' + a + 'h_updates', be.zeros((nout, nout)))
            setattr(self, 'b_' + a + '_updates', be.zeros((nout, 1)))
            self.backend.end()

        # If this isn't initialized correctly, get NaNs pretty quickly.
        be.add(be.zeros((nout, 1)), 1, self.b_i)   # sigmoid(1) opens the gate
        # +5 following clockwork RNN paper "to encourage long term memory"
        be.add(be.zeros((nout, 1)), -1, self.b_f)  # sigmoid(-1) closes gate.
        be.add(be.zeros((nout, 1)), 1, self.b_o)   # sigmoid(1) open
        self.b_c = be.zeros((nout, 1))  # no need to be messed with

        # and for higher up entities in the LSTM cell.
        self.c_t = [be.zeros(net_sze) for k in range(unrolls)]
        self.c_phi = [be.zeros(net_sze) for k in range(unrolls)]
        self.c_phip = [be.zeros(net_sze) for k in range(unrolls)]
        self.output_list = [be.zeros(net_sze) for k in range(unrolls)]

        # pre-allocate preactivation buffers
        self.temp_x = [be.zeros(net_sze) for k in range(unrolls)]
        self.temp_h = [be.zeros(net_sze) for k in range(unrolls)]

        self.learning_rule.allocate_state_lstm(self.Wix_updates,
                                               self.Wih_updates,
                                               self.b_i_updates)

        self.deltas = be.zeros((nout, batch_size))  # hidden bprop error
        self.cerror = be.zeros((nout, batch_size))  # cell bprop error

        self.temp_t = 0

    def fprop(self, y, inputs, tau, cell):
        """
        Forward pass for the google-style LSTM cell with forget gates, no
        peepholes.

        Inputs:
            y:      input from prev. time step (eg. one batch of (64, 50) size)
            inputs: input from data (eg. one batch of (128, 50) size)
            (tau):  unrolling step for BPTT
            cell:   state of memory cell from prev. time step (shape as y)

        Outputs:
            self.c_t:         cell activity
            self.output_list: hidden activity

        In math notiation, forward pass:
            i_t = s(Wix*x + Wih*h +b_i)
            f_t = s(Wpx*x + Wfh*h +b_f)
            o_t = s(Wox*x + Woh*h +b_o)
            g_t = s(Wcx*x + Wch*h +b_c)
            c_t = f_t .* c_t-1 + i_t .* g_t
            h_t = o_t .* phi(c_t)
            ------ output layer -----
            y_t = s(W_yh * h_t)
            e_t = xEnt(y, t)

        The values are computed and stored for all unrolls so they can be
        used in bprop. [TODO] check for redundant buffers
        """
        be = self.backend  # shorthand
        phi = self.activation  # tanh
        sig = self.gate_activation  # logistic

        # input gate
        be.fprop_fc(self.temp_x[tau], inputs, self.Wix)
        be.fprop_fc(self.temp_h[tau], y, self.Wih)
        be.add(self.temp_x[tau], self.temp_h[tau], self.net_i[tau])
        be.add(self.net_i[tau], self.b_i, self.net_i[tau])
        sig.apply_both(be, self.net_i[tau], self.i_t[tau])

        # forget gate
        be.fprop_fc(self.temp_x[tau], inputs, self.Wfx)
        be.fprop_fc(self.temp_h[tau], y, self.Wfh)
        be.add(self.temp_x[tau], self.temp_h[tau], self.net_f[tau])
        be.add(self.net_f[tau], self.b_f, self.net_f[tau])
        sig.apply_both(be, self.net_f[tau], self.f_t[tau])

        # output gate
        be.fprop_fc(self.temp_x[tau], inputs, self.Wox)
        be.fprop_fc(self.temp_h[tau], y, self.Woh)
        be.add(self.temp_x[tau], self.temp_h[tau], self.net_o[tau])
        be.add(self.net_o[tau], self.b_o, self.net_o[tau])
        sig.apply_both(be, self.net_o[tau], self.o_t[tau])

        # classic RNN cell
        be.fprop_fc(self.temp_x[tau], inputs, self.Wcx)
        be.fprop_fc(self.temp_h[tau], y, self.Wch)
        be.add(self.temp_x[tau], self.temp_h[tau], self.net_g[tau])
        be.add(self.net_g[tau], self.b_c, self.net_g[tau])
        phi.apply_both(be, self.net_g[tau], self.g_t[tau])

        # combine the parts and compute output.
        be.multiply(self.f_t[tau], cell, self.c_t[tau])
        be.multiply(self.i_t[tau], self.g_t[tau], self.c_phip[tau])
        be.add(self.c_t[tau], self.c_phip[tau], self.c_t[tau])
        self.c_phip[tau] = be.copy(self.c_t[tau])

        phi.apply_both(be, self.c_phip[tau], self.c_phi[tau])
        be.multiply(self.o_t[tau], self.c_phi[tau], self.output_list[tau])

    def bprop(self, error_h, error_c, inputs, tau_tot, tau, numgrad=False):
        """
        For LSTM, inject h-error and c-error, get 8 w's and h, c out. It's
        more complicated than bprop thorugh a standard layer mostly because
        we have two outputs that we inject errors into, each leading to an
        error on the two inputs (4 errors total), and each of the weight
        updates has a contribution from the error to the cell and the hidden.


        Inputs:
            error_h2: error injected into hidden
            error_c2: error injected directly into cell

        Outputs:
            error_h1: from h2 and c2: dh2/dh1 + dc2/dh1
                                      existing  new
            error_c1: from h2 and c2: dh2/dc1 + dc2/dc1
                                      new       new

        [TODO] Two new terms to compute!

        Basic derivation
            In math, backward pass:
                de_dJ = d/dJ CE(y,t)
                dy_dJ = d/dJ sigm(wyh*h)
                ------ hidden layer -----
                dh_dJ = d/dJ o .* tanh(c)
                dp_dJ = d/dJ phi(c)
                dc_dJ = d/dJ (f.*c_ + i.*g)
                di_dJ = d/dJ s(wix*x+wih*h+b)
                df_dJ = d/dJ s(wfx*x+wfh*h+b)
                do_dJ = d/dJ s(wcx*x+wch*h+b)
                dg_dJ = d/dJ s(wcx*x+wch*h+b)

        Over multiple time-steps, deltas feeds back in as error.
        [TODO] Currently using a bunch of if statements to catch propagating
        into outputs[-1], which should not wrap but be 0.
        """
        be = self.backend

        # 1. allocate buffers -  these are for a single pass through the cell,
        # the wix_updates etc. accumulate over the loop.
        # [TODO] allocate in init, call them self.dh_dw['ix']

        di_dh1 = be.zeros((self.nout, self.batch_size))
        df_dh1 = be.copy(di_dh1)
        do_dh1 = be.copy(di_dh1)
        dg_dh1 = be.copy(di_dh1)
        hherror = be.copy(di_dh1)
        hcerror = be.copy(di_dh1)
        cherror = be.copy(di_dh1)
        ccerror = be.copy(di_dh1)

        # [todo] need only two temp buffers here
        dh_dwix = be.zeros((self.nout, self.nin))
        dh_dwfx = be.copy(dh_dwix)
        dh_dwox = be.copy(dh_dwix)
        dh_dwcx = be.copy(dh_dwix)
        dh_dwih = be.zeros((self.nout, self.nout))
        dh_dwfh = be.copy(dh_dwih)
        dh_dwoh = be.copy(dh_dwih)
        dh_dwch = be.copy(dh_dwih)

        dc_di_dh1 = be.copy(di_dh1)
        dc_df_dh1 = be.copy(di_dh1)
        dc_dg_dh1 = be.copy(di_dh1)

        """--------------------------
        PART 1: original dh2/dh1 terms
        --------------------------"""
        temp = be.zeros((self.nout, self.batch_size))
        temp_sum = be.empty((self.nout, 1))
        # a. Input gate
        # temp = error_h * self.o_t[tau] * self.c_phip[tau] * self.g_t[tau] \
        #        * self.net_i[tau]
        be.multiply(error_h, self.o_t[tau], temp)
        be.multiply(self.c_phip[tau], temp, temp)
        be.multiply(self.g_t[tau], temp, temp)
        be.multiply(self.net_i[tau], temp, temp)
        be.bprop_fc(out=di_dh1, weights=self.Wih, deltas=temp)
        be.update_fc(out=dh_dwix,
                     inputs=inputs[tau*128:(tau+1)*128, :],
                     deltas=temp)
        be.update_fc(out=dh_dwih,
                     inputs=self.output_list[tau - 1],
                     deltas=temp)
        be.add(self.Wix_updates, dh_dwix, self.Wix_updates)
        if (tau > 0):
            be.add(self.Wih_updates, dh_dwih, self.Wih_updates)
        be.sum(temp, 1, temp_sum)
        be.add(self.b_i_updates, temp_sum, self.b_i_updates)

        # b. forget gate
        # temp = error_h * self.o_t[tau] * self.c_phip[tau] * self.c_t[tau-1] \
        #        * self.net_f[tau]
        be.multiply(error_h, self.o_t[tau], temp)
        be.multiply(self.c_phip[tau], temp, temp)
        be.multiply(self.c_t[tau-1], temp, temp)
        be.multiply(self.net_f[tau], temp, temp)
        be.bprop_fc(out=df_dh1, weights=self.Wfh, deltas=temp)
        be.update_fc(out=dh_dwfx,
                     inputs=inputs[tau*128:(tau+1)*128, :],
                     deltas=temp)
        be.update_fc(out=dh_dwfh,
                     inputs=self.output_list[tau - 1],
                     deltas=temp)
        be.add(self.Wfx_updates, dh_dwfx, self.Wfx_updates)
        if (tau > 0):
            be.add(self.Wfh_updates, dh_dwfh, self.Wfh_updates)
        be.sum(temp, 1, temp_sum)
        be.add(self.b_f_updates, temp_sum, self.b_f_updates)

        # c. output gate
        # temp = error_h * self.c_phi[tau] * self.net_o[tau]
        be.multiply(error_h, self.c_phi[tau], temp)
        be.multiply(self.net_o[tau], temp, temp)
        be.bprop_fc(out=do_dh1, weights=self.Woh, deltas=temp)
        be.update_fc(out=dh_dwox,
                     inputs=inputs[tau*128:(tau+1)*128, :],
                     deltas=temp)
        be.update_fc(out=dh_dwoh,
                     inputs=self.output_list[tau - 1],
                     deltas=temp)
        be.add(self.Wox_updates, dh_dwox, self.Wox_updates)
        if (tau > 0):
            be.add(self.Woh_updates, dh_dwoh, self.Woh_updates)
        be.sum(temp, 1, temp_sum)
        be.add(self.b_o_updates, temp_sum, self.b_o_updates)

        # d. cell
        # temp = error_h * self.o_t[tau] * self.c_phip[tau] * self.i_t[tau] \
        #        * self.net_g[tau]
        be.multiply(error_h, self.o_t[tau], temp)
        be.multiply(self.c_phip[tau], temp, temp)
        be.multiply(self.i_t[tau], temp, temp)
        be.multiply(self.net_g[tau], temp, temp)
        be.bprop_fc(out=dg_dh1, weights=self.Wch, deltas=temp)
        be.update_fc(out=dh_dwcx,
                     inputs=inputs[tau*128:(tau+1)*128, :],
                     deltas=temp)
        be.update_fc(out=dh_dwch,
                     inputs=self.output_list[tau - 1],
                     deltas=temp)
        be.add(self.Wcx_updates, dh_dwcx, self.Wcx_updates)
        if (tau > 0):
            be.add(self.Wch_updates, dh_dwch, self.Wch_updates)
        be.sum(temp, 1, temp_sum)
        be.add(self.b_c_updates, temp_sum, self.b_c_updates)

        # e. collect terms
        be.add(di_dh1, df_dh1, hherror)
        be.add(do_dh1, hherror, hherror)
        be.add(dg_dh1, hherror, hherror)

        # used for num grad checks
        ttemp1i = dh_dwih[12, 55].asnumpyarray()
        ttemp1f = dh_dwfh[12, 55].asnumpyarray()
        ttemp1o = dh_dwoh[12, 55].asnumpyarray()
        ttemp1c = dh_dwch[12, 55].asnumpyarray()

        """ --------------------------
        PART 2: New dc2/dc1 dc2/dh1 and dh2/dc1 terms
        ---------------------------"""

        # dc2/dh1 terms:
        # input gate
        # temp = error_c * self.g_t[tau] * self.net_i[tau]
        be.multiply(error_c, self.g_t[tau], temp)
        be.multiply(self.net_i[tau], temp, temp)
        be.bprop_fc(out=dc_di_dh1, weights=self.Wih, deltas=temp)
        be.update_fc(out=dh_dwix,
                     inputs=inputs[tau*128:(tau+1)*128, :],
                     deltas=temp)
        be.update_fc(out=dh_dwih,
                     inputs=self.output_list[tau - 1],
                     deltas=temp)
        be.add(self.Wix_updates, dh_dwix, self.Wix_updates)
        if (tau > 0):
            be.add(self.Wih_updates, dh_dwih, self.Wih_updates)
        be.sum(temp, 1, temp_sum)
        be.add(self.b_i_updates, temp_sum, self.b_i_updates)

        # forget gate
        # temp = error_c * self.c_t[tau-1] * self.net_f[tau]
        be.multiply(error_c, self.c_t[tau-1], temp)
        be.multiply(self.net_f[tau], temp, temp)
        be.bprop_fc(out=dc_df_dh1, weights=self.Wfh, deltas=temp)
        be.update_fc(out=dh_dwfx,
                     inputs=inputs[tau*128:(tau+1)*128, :],
                     deltas=temp)
        be.update_fc(out=dh_dwfh,
                     inputs=self.output_list[tau - 1],
                     deltas=temp)
        be.add(self.Wfx_updates, dh_dwfx, self.Wfx_updates)
        if (tau > 0):
            be.add(self.Wfh_updates, dh_dwfh, self.Wfh_updates)
        be.sum(temp, 1, temp_sum)
        be.add(self.b_f_updates, temp_sum, self.b_f_updates)

        # cell
        # temp = error_c * self.i_t[tau] * self.net_g[tau]
        be.multiply(error_c, self.i_t[tau], temp)
        be.multiply(self.net_g[tau], temp, temp)
        be.bprop_fc(out=dc_dg_dh1, weights=self.Wch, deltas=temp)
        be.update_fc(out=dh_dwcx,
                     inputs=inputs[tau*128:(tau+1)*128, :],
                     deltas=temp)
        be.update_fc(out=dh_dwch,
                     inputs=self.output_list[tau - 1],
                     deltas=temp)
        be.add(self.Wcx_updates, dh_dwcx, self.Wcx_updates)
        if (tau > 0):
            be.add(self.Wch_updates, dh_dwch, self.Wch_updates)
        be.sum(temp, 1, temp_sum)
        be.add(self.b_c_updates, temp_sum, self.b_c_updates)

        be.add(dc_di_dh1, dc_df_dh1, cherror)
        be.add(cherror, dc_dg_dh1, cherror)

        # dh2/dc1 term:
        # hcerror = error_h * self.o_t[tau] * self.c_phip[tau] * self.f_t[tau]
        be.multiply(error_h, self.o_t[tau], hcerror)
        be.multiply(self.c_phip[tau], hcerror, hcerror)
        be.multiply(self.f_t[tau], hcerror, hcerror)

        # dc2/dc1 term:
        # ccerror = error_c * self.f_t[tau]
        be.multiply(error_c, self.f_t[tau], ccerror)

        # wrap up:
        be.add(hherror, cherror, self.deltas)
        be.add(ccerror, hcerror, self.cerror)

        if numgrad is "lstm_ih":
            ttemp2i = dh_dwih[12, 55].asnumpyarray()
            logger.info("layer.LSTM.bprop: analytic dh_dwih[%d]= %e + %e = %e",
                        tau, ttemp1i, ttemp2i, ttemp1i + ttemp2i)
        if numgrad is "lstm_fh":
            ttemp2f = dh_dwfh[12, 55].asnumpyarray()
            logger.info("layer.LSTM.bprop: analytic dh_dwfh[%d]= %e + %e = %e",
                        tau, ttemp1f, ttemp2f, ttemp1f + ttemp2f)
        if numgrad is "lstm_oh":
            ttemp2o = dh_dwoh[12, 55].asnumpyarray()
            logger.info("layer.LSTM.bprop: analytic dh_dwoh[%d]= %e + %e = %e",
                        tau, ttemp1o, ttemp2o, ttemp1o + ttemp2o)
        if numgrad is "lstm_ch":
            ttemp2c = dh_dwch[12, 55].asnumpyarray()
            logger.info("layer.LSTM.bprop: analytic dh_dwch[%d]= %e + %e = %e",
                        tau, ttemp1c, ttemp2c, ttemp1c + ttemp2c)

    def update(self, epoch):
        """
        Need to think of something new here, can't have a new rule for each
        of the matrices. Why does apply_rule not take different weights?
        """
        self.learning_rule.apply_rule_lstm(
            (self.Wix, self.Wfx, self.Wox, self.Wcx,
             self.Wih, self.Wfh, self.Woh, self.Wch,
             self.b_i, self.b_f, self.b_o, self.b_c),
            (self.Wix_updates, self.Wfx_updates,
             self.Wox_updates, self.Wcx_updates,
             self.Wih_updates, self.Wfh_updates,
             self.Woh_updates, self.Wch_updates,
             self.b_i_updates, self.b_f_updates,
             self.b_o_updates, self.b_c_updates),
            epoch)


class RecurrentHiddenLayer(Layer):

    """
    Derived from Layer. In addition to the lists[tau] outlined for
    RecurrentOutputLayer, the fprop is getting input from two weight matrices,
    one connected to the input and one connected to the previous hidden state.
    """

    def __init__(self, name, backend, batch_size, pos, nin, nout, unrolls,
                 activation, weight_init, weight_init_rec, learning_rule,
                 weight_dtype=None, delta_dtype=None, updates_dtype=None,
                 pre_act_dtype=None, output_dtype=None, deltas_dtype=None):
        super(RecurrentHiddenLayer, self).__init__(name, backend, batch_size,
                                                   pos, nin, nout, weight_init,
                                                   learning_rule,
                                                   activation=activation)
        self.weights_rec = weight_init_rec.generate((nout, nout), weight_dtype)
        self.pre_act_list = [self.backend.zeros((nout, batch_size),
                                                pre_act_dtype)
                             for k in range(unrolls)]
        self.output_list = [self.backend.zeros((nout, batch_size),
                                               output_dtype)
                            for k in range(unrolls)]
        self.updates_rec = self.backend.zeros((nout, nout))
        self.temp_rec = self.backend.zeros((nout, nout))
        self.temp_in = self.backend.zeros((nout, nin))
        self.learning_rule.allocate_state_rec(self.updates_rec)

        self.deltas = backend.zeros((nout, batch_size))

    def fprop(self, y, inputs, tau, cell=None):
        z1 = self.backend.zeros(self.pre_act_list[tau].shape)
        z2 = self.backend.zeros(self.pre_act_list[tau].shape)
        self.backend.fprop_fc(z1, y, self.weights_rec)
        self.backend.fprop_fc(z2, inputs, self.weights)
        self.backend.add(z1, z2, self.pre_act_list[tau])
        if self.activation is not None:
            self.activation.apply_both(self.backend,
                                       self.pre_act_list[tau],
                                       self.output_list[tau])
        else:
            raise AttributeError("Urs is not cool with your missing "
                                 "activation function")

    def bprop(self, error, error_c, inputs, tau, t, numgrad=False):
        """
        Backpropagation for recurrent hidden layers.

        Notes:
            * This function has been refactored
            * [done] remove duplicate code
            * [done] remove the loop altogether.
            * [todo] If the if statement can't be supported, revert to
              duplicated code
            * Not sure why tau is passed but not used. Not that this is
              called for decrementing t.
        """
        self.backend.multiply(error, self.pre_act_list[t], out=error)
        if (t > 0):  # can move down or just compute (but it's not used)
            # compute error (apply prev. delta)
            self.backend.bprop_fc(out=self.deltas,  # output for next iteration
                                  weights=self.weights_rec,
                                  deltas=error)

        # input weight update (apply curr. delta)
        self.backend.update_fc(out=self.temp_in,
                               inputs=inputs[t*128:(t+1)*128, :],
                               deltas=error)
        self.backend.add(self.weight_updates, self.temp_in,
                         self.weight_updates)

        if (t > 0):
            # recurrent weight update (apply prev. delta)
            self.backend.update_fc(out=self.temp_rec,
                                   inputs=self.output_list[t - 1],  # avoid t=0
                                   deltas=error)
            self.backend.add(self.updates_rec, self.temp_rec, self.updates_rec)
        if numgrad is "input":
            logger.info("RecurrentHiddenLayer.bprop inc in %e",
                        self.temp_in[12, 110].asnumpyarray())
        if numgrad is "rec":
            logger.info("RecurrentHiddenLayer.bprop inc rec %e",
                        self.temp_rec[12, 63].asnumpyarray())

    def update(self, epoch):
        self.learning_rule.apply_rule(self.params, self.updates, epoch)
        self.learning_rule.apply_rule_rec(self.weights_rec,
                                          self.updates_rec, epoch)


class BranchLayer(YAMLable):

    """
    Branch layer is composed of a list of other layers
    during fprop, it concatenates the component outputs and passes it on
    during bprop, it splits the backward errors into the components and
        accumulates into a common deltas
    """

    def __init__(self, name, backend, batch_size, pos, nin, sublayers,
                 output_dtype=None, deltas_dtype=None, prev_names=[]):
        self.name = name
        self.backend = backend
        self.nin = nin
        self.nout = 0
        self.sublayers = sublayers
        self.nsublayers = len(self.sublayers)
        self.startidx = [0] * len(self.sublayers)
        self.endidx = [0] * len(self.sublayers)
        self.prev_names = prev_names
        self.batch_size = batch_size

        for i in range(self.nsublayers):
            self.backend.begin()
            self.nout += self.sublayers[i].nout
            self.endidx[i] = self.nout
            if i > 0:
                self.startidx[i] = (self.startidx[i - 1] +
                                    self.sublayers[i - 1].nout)
            self.backend.end()

        self.output = backend.empty((self.nout, batch_size), output_dtype)
        self.pos = pos
        if pos > 0:
            self.deltas = backend.empty((nin, batch_size), deltas_dtype)

    def fprop(self, inputs):
        for (sublayer, s_idx, e_idx) in zip(self.sublayers,
                                            self.startidx, self.endidx):
            self.backend.begin()
            sublayer.fprop(inputs)
            self.output[s_idx:e_idx] = sublayer.output
            self.backend.end()

    def bprop(self, error, inputs):
        for (sublayer, s_idx, e_idx) in zip(self.sublayers,
                                            self.startidx, self.endidx):
            self.backend.begin()
            sublayer.bprop(error[s_idx:e_idx], inputs)
            self.backend.end()

        if self.pos > 0:
            self.deltas[:] = 0.0
            for sublayer in self.sublayers:
                self.backend.begin()
                self.backend.add(self.deltas, sublayer.deltas, out=self.deltas)
                self.backend.end()

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        for sublayer in self.sublayers:
            self.backend.begin()
            sublayer.set_train_mode(mode)
            self.backend.end()


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
                 output_dtype=None, deltas_dtype=None, prev_names=[]):
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
            self.deltas = backend.empty((nin, batch_size), deltas_dtype)
        self.prev_names = prev_names

    def fprop(self, inputs):
        if (self.train_mode):
            self.backend.fill_uniform_thresh(self.keepmask, self.keep)
            self.backend.multiply(self.keepmask, self.backend.wrap(self.keep),
                                  out=self.keepmask)
            self.backend.multiply(inputs, self.keepmask, out=self.output)
        else:
            self.backend.multiply(inputs, self.keep, out=self.output)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.multiply(error, self.keepmask, out=self.deltas)

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        self.train_mode = mode


class RBMLayer(Layer):

    """
    CD1 training layer for RBM
    """

    def __init__(self, name, backend, batch_size, pos, nin,
                 nout, activation, weight_init, learning_rule, prev_names=[]):
        super(RBMLayer, self).__init__(name, backend, batch_size, pos,
                                       nin, nout, weight_init,
                                       learning_rule, activation=activation,
                                       prev_names=prev_names)
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
        self.output = backend.empty((self.nin, batch_size))

    def positive(self, inputs):
        """
        Positive / upward pass of the CD1 RBM

        Arguments:
           inputs (neon.datasets.dataset.Dataset): dataset upon which
                                                      to operate
        """
        self.backend.dot(self.weights, inputs, out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_plus)
        self.backend.dot(self.p_hid_plus, inputs.transpose(), out=self.p_plus)
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
        self.backend.dot(self.weights.transpose(), self.s_hid_plus,
                         out=self.neg_pre_act)
        self.activation.apply_function(self.backend, self.neg_pre_act,
                                       self.x_minus)
        self.backend.dot(self.weights, self.x_minus, out=self.pre_act)
        self.activation.apply_function(self.backend, self.pre_act,
                                       self.p_hid_minus)
        self.output[:] = self.x_minus

    def update(self, epoch):
        """
        CD1 weight update

        Arguments:
            epoch: not used, for future compatibility
        """
        self.backend.dot(self.p_hid_minus, self.x_minus.transpose(),
                         out=self.p_minus)
        self.backend.subtract(self.p_plus, self.p_minus, out=self.diff)
        self.learning_rule.apply_rule([self.weights], [self.diff], epoch)


class LocalLayer(YAMLable):

    """
    Base class for locally connected layers.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, pooling=False,
                 activation=None, pad=0, prev_names=[]):
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
        self.fpsize = self.fheight * self.fwidth
        self.stride = stride
        self.learning_rule = learning_rule
        self.prev_names = prev_names

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
            self.deltas = backend.empty((self.nin, batch_size))
            self.deltasbuf = backend.empty((self.ifmsize, batch_size * nifm))

        self.fsize = nifm * self.fheight * self.fwidth
        ofmstarts = backend.array(range(0, (self.ofmsize * nofm),
                                        self.ofmsize))
        self.ofmlocs = backend.empty((self.ofmsize, nofm), dtype='int32')
        for dst in range(self.ofmsize):
            self.backend.begin()
            backend.add(ofmstarts, dst, self.ofmlocs[dst])
            self.backend.end()

        # Figure out the connections with the previous layer.
        if not isinstance(backend, CPU):
            self.rlinks = []
            if pooling is True:
                self.links = []
                self.outputbuf = []
                if pos > 0:
                    self.deltasbuf = []
            else:
                self.links = []
        else:
            if pooling is True:
                self.links = backend.empty(
                    (self.ofmsize, fshape[0] * fshape[1]), dtype='i32')
                self.outputbuf = backend.empty(
                    (self.ofmsize, batch_size * nifm))
            else:
                self.links = backend.empty(
                    (self.ofmsize, self.fsize), dtype='i32')

            # This variable tracks the top left corner of the receptive field.
            src = 0
            for dst in range(self.ofmsize):
                self.backend.begin()
                # Collect the column indices for the
                # entire receptive field.
                colinds = []
                for row in xrange(self.fheight):
                    self.backend.begin()
                    start = src + row * self.ifmwidth
                    colinds += range(start, start + self.fwidth)
                    self.backend.end()
                fminds = colinds[:]
                if pooling is False:
                    for ifm in range(1, nifm):
                        self.backend.begin()
                        colinds += [x + ifm * self.ifmsize for x in fminds]
                        self.backend.end()

                expr1 = (src % self.ifmwidth + self.fwidth + stride)
                if expr1 <= self.ifmwidth:
                    # Slide the filter to the right by the stride value.
                    src += stride
                else:
                    # We hit the right edge of the input image.
                    # Shift the filter down by one stride.
                    src += stride * self.ifmwidth - src % self.ifmwidth
                    assert src % self.ifmwidth == 0
                self.links[dst, :] = backend.array(colinds, dtype='i32')
                self.backend.end()
            self.rlinks = self.links.asnumpyarray()

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
                 activation=None, pad=0, prev_names=[]):
        self.name = name
        self.backend = backend
        self.activation = activation
        self.pad = pad
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmshape = ifmshape
        self.fshape = fshape
        self.fheight, self.fwidth = fshape
        self.fpsize = self.fheight * self.fwidth
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
        #    self.deltas = backend.empty((batch_size, self.nin), dtype=dtype)
        self.nifm = nifm
        self.nofm = nofm
        self.fsize = nifm * self.fheight * self.fwidth
        self.stride = stride
        self.pooling = pooling
        self.prev_names = prev_names

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
            self.deltas = self.backend.empty((self.nin, self.batch_size))

        ofmstarts = self.backend.array(range(0, (self.ofmsize * self.nofm),
                                             self.ofmsize))

        self.ofmlocs = self.backend.empty((self.ofmsize, self.nofm),
                                          dtype='int32')
        for dst in range(self.ofmsize):
            self.backend.begin()
            self.backend.add(ofmstarts, dst, self.ofmlocs[dst])
            self.backend.end()

        # stores the flattened px location across
        # ofm in columns

        # Figure out the connections with the previous layer.
        if self.pooling is True:
            self.links = self.backend.empty(
                (self.ofmsize, self.fshape[0] * self.fshape[1]), dtype='int32')
            self.outputbuf = self.backend.empty((self.ofmsize,
                                                 self.batch_size * self.nifm))
            if self.pos > 0:
                self.deltasbuf = self.backend.empty(
                    (self.ifmsize, self.batch_size * self.nifm))
        else:
            self.links = self.backend.empty(
                (self.ofmsize, self.fsize), dtype='int32')
        # This variable tracks the top left corner of the receptive field.
        src = 0
        for dst in range(self.ofmsize):
            self.backend.begin()
            # Collect the column indices for the
            # entire receptive field.
            colinds = []
            for row in range(self.fheight):
                self.backend.begin()
                start = src + row * self.ifmwidth
                colinds += range(start, start + self.fwidth)
                self.backend.end()
            fminds = colinds[:]
            if self.pooling is False:
                for ifm in range(1, self.nifm):
                    self.backend.begin()
                    colinds += [x + ifm * self.ifmsize for x in fminds]
                    self.backend.end()

            if (src % self.ifmwidth + self.fwidth + self.stride) <= (
                    self.ifmwidth):
                # Slide the filter to the right by the stride value.
                src += self.stride
            else:
                # We hit the right edge of the input image.
                # Shift the filter down by one stride.
                src += self.stride * self.ifmwidth - src % self.ifmwidth
                assert src % self.ifmwidth == 0
            self.links[dst] = self.backend.array(colinds)
            self.backend.end()
        self.rlinks = self.links.asnumpyarray()

        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.empty((self.nout, self.batch_size))


class ConvLayer(LocalLayer):

    """
    Convolutional layer.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, weight_init, activation=None,
                 pad=0, prev_names=[]):
        if pad != 0 and isinstance(backend, CPU):
            raise NotImplementedError('pad != 0, for CPU backend in ConvLayer')
        super(ConvLayer, self).__init__(name, backend, batch_size, pos,
                                        learning_rule, nifm, nofm,
                                        ifmshape, fshape, stride,
                                        activation=activation,
                                        pad=pad, prev_names=prev_names)
        self.nout = self.ofmsize * nofm
        self.weights = weight_init.generate((self.fsize, nofm))
        self.output = backend.empty((self.nout, batch_size))
        self.weight_updates = backend.empty(self.weights.shape)
        if not isinstance(backend, CPU):
            self.prodbuf = []
            self.bpropbuf = []
            self.updatebuf = []
        else:
            self.prodbuf = backend.empty((nofm, batch_size))
            self.bpropbuf = backend.empty((self.fsize, batch_size))
            self.updatebuf = backend.empty(self.weights.shape)

        if activation is not None:
            self.pre_act = backend.empty((self.nout, batch_size))
        else:
            self.pre_act = self.output

        # start bias related
        self.use_biases = 'bias_init' in weight_init.__dict__
        if self.use_biases:
            self.biases = self.backend.empty((self.nout, 1))
            self.backend.fill(self.biases, weight_init.bias_init)
            self.bias_updates = self.backend.empty(self.biases.shape)
            self.params = [self.weights, self.biases]
            self.updates = [self.weight_updates, self.bias_updates]
        else:
            self.params = [self.weights]
            self.updates = [self.weight_updates]
        # end bias related
        self.learning_rule.allocate_state(self.updates)

    def __str__(self):
        temp = self.backend.empty((1, 1))
        return ("ConvLayer %s: %d ifms, %d filters, "
                "utilizing %s backend\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nifm, self.nofm,
                 self.backend.__class__.__name__,
                 float(self.backend.mean(self.weights, axes=None,
                                         out=temp).asnumpyarray()),
                 float(self.backend.min(self.weights, axes=None,
                                        out=temp).asnumpyarray()),
                 float(self.backend.max(self.weights, axes=None,
                                        out=temp).asnumpyarray())))

    def fprop(self, inputs):
        self.backend.fprop_conv(out=self.pre_act, inputs=inputs,
                                weights=self.weights, ofmshape=self.ofmshape,
                                ofmsize=self.ofmsize,
                                ofmlocs=self.ofmlocs, ifmshape=self.ifmshape,
                                links=self.rlinks, nifm=self.nifm,
                                padding=self.pad, stride=self.stride,
                                ngroups=1, fpropbuf=self.prodbuf)
        if self.use_biases is True:
            self.backend.add(self.pre_act, self.biases, out=self.pre_act)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs):
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)
        if self.pos > 0:
            self.backend.bprop_conv(out=self.deltas, weights=self.weights,
                                    deltas=error, ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=self.ofmlocs,
                                    ifmshape=self.ifmshape, links=self.links,
                                    padding=self.pad, stride=self.stride,
                                    nifm=self.nifm, ngroups=1,
                                    bpropbuf=self.bpropbuf)
        self.backend.update_conv(out=self.weight_updates, inputs=inputs,
                                 weights=self.weights, deltas=error,
                                 ofmshape=self.ofmshape,
                                 ofmsize=self.ofmsize,
                                 ofmlocs=self.ofmlocs,
                                 ifmshape=self.ifmshape, links=self.links,
                                 nifm=self.nifm, padding=self.pad,
                                 stride=self.stride, ngroups=1,
                                 fwidth=self.fwidth, updatebuf=self.updatebuf)
        if self.use_biases is True:
            self.backend.sum(error, axis=1, out=self.bias_updates)

    def update(self, epoch):
        self.learning_rule.apply_rule(self.params, self.updates, epoch)


class ConvLayerDist(LocalLayerDist, ConvLayer):

    """
    Distributed convolutional layer.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule, nifm,
                 nofm, ifmshape, fshape, stride, weight_init, activation=None,
                 pad=0, prev_names=[]):
        if pad != 0:
            raise NotImplementedError('Pad != 0, for ConvLayerDist')
        super(ConvLayerDist, self).__init__(name, backend, batch_size, pos,
                                            learning_rule, nifm, nofm,
                                            ifmshape, fshape, stride,
                                            activation=activation, pad=pad,
                                            prev_names=prev_names)
        self.use_biases = 'bias_init' in weight_init.__dict__
        if self.use_biases:
            raise NotImplementedError('TODO')
        self.nout = self.ofmsize * nofm
        self.weights = weight_init.generate((self.fsize, nofm))
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
            self.backend.bprop_conv(out=self.deltas, weights=self.weights,
                                    deltas=error, ofmshape=self.ofmshape,
                                    ofmlocs=self.ofmlocs,
                                    ifmshape=self.ifmshape, links=self.links,
                                    padding=0, stride=self.stride,
                                    nifm=self.nifm, ngroups=1,
                                    bpropbuf=self.bpropbuf)
        # accumulate updates across tiles for all filters
        # if want to keep weights unshared across nodes, could not do the
        # transfers here
        self.updates._tensor = MPI.COMM_WORLD.reduce(
            self.updates.asnumpyarray(), op=MPI.SUM, root=0)
        self.updates._tensor = MPI.COMM_WORLD.bcast(
            self.updates.asnumpyarray())
        self.backend.update_conv(out=self.updates, inputs=inputs,
                                 weights=self.weights, deltas=error,
                                 ofmshape=self.ofmshape,
                                 ofmsize=self.ofmsize,
                                 ofmlocs=self.ofmlocs,
                                 ifmshape=self.ifmshape, links=self.links,
                                 nifm=self.nifm, padding=0, stride=self.stride,
                                 ngroups=1, fwidth=self.fwidth,
                                 updatebuf=self.updatebuf)

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
                 pretraining, sparsity, tied_weights, prev_names=[]):
        super(LocalFilteringLayer, self).__init__(name, backend, batch_size,
                                                  pos, learning_rule,
                                                  nifm, nofm, ifmshape, fshape,
                                                  stride,
                                                  prev_names=prev_names)
        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nout = self.ofmsize * nofm
        self.output = backend.empty((self.nout, batch_size))
        self.weights = weight_init.generate((self.nout, self.fsize))

        self.normalize_weights(self.weights)
        self.updates = backend.empty(self.weights.shape)
        self.prodbuf = backend.empty((nofm, batch_size))
        self.bpropbuf = backend.empty((self.fsize, batch_size))
        self.updatebuf = backend.empty((nofm, self.fsize))
        self.learning_rule = learning_rule

        self.learning_rule.allocate_state([self.updates])
        if pretraining is True:
            self.sparsity = sparsity
            self.tied_weights = tied_weights

    def __str__(self):
        temp = self.backend.empty((1, 1))
        return ("LocalFilteringLayer %s: %d ifms, "
                "utilizing %s backend\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nifm,
                 self.backend.__class__.__name__,
                 float(self.backend.mean(self.weights, axes=None,
                                         out=temp).asnumpyarray()),
                 float(self.backend.min(self.weights, axes=None,
                                        out=temp).asnumpyarray()),
                 float(self.backend.max(self.weights, axes=None,
                                        out=temp).asnumpyarray())))

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
        self.backend.divide(error, inputs.shape[1], out=error)
        self.defilter.bprop(error, self.output)
        self.defilter.update(epoch)
        # Now backward propagate the gradient of the output of the
        # pooling layer.
        error = self.backend.ones(self.pooling.output.shape)
        self.backend.multiply(error, self.sparsity / inputs.shape[1], error)
        self.pooling.bprop(error, self.output)
        # Aggregate the errors from both layers before back propagating
        # through the current layer.
        deltas = self.backend.empty(self.defilter.deltas.shape)
        self.backend.add(self.defilter.deltas, self.pooling.deltas, deltas)
        self.bprop(deltas, inputs)
        self.update(epoch)
        rcost = cost.apply_function(inputs)
        spcost = self.backend.empty((1, 1))
        self.backend.sum(self.pooling.output, axes=None, out=spcost)
        self.backend.multiply(spcost, self.sparsity, spcost)
        return rcost, spcost

    def fprop(self, inputs):
        for dst in range(self.ofmsize):
            self.backend.begin()
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
            self.output[self.ofmlocs[dst]] = self.prodbuf
            self.backend.end()

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.deltas.fill(0)
            for dst in range(self.ofmsize):
                self.backend.begin()
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
                                 self.deltas.take(rflinks, axis=0),
                                 out=self.bpropbuf)
                self.deltas[rflinks] = self.bpropbuf
                self.backend.end()

        for dst in range(self.ofmsize):
            self.backend.begin()
            rflinks = self.rlinks[dst]
            delta_slice = error.take(self.ofmlocs[dst], axis=0)
            self.backend.dot(delta_slice,
                             inputs.take(rflinks, axis=0).transpose(),
                             out=self.updatebuf)
            self.updates[self.ofmlocs[dst]] = self.updatebuf
            self.backend.end()

    def update(self, epoch):
        self.learning_rule.apply_rule([self.weights], [self.updates], epoch)
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
        # self.weights = self.weight_init.generate((self.nout, self.fsize),
        #                                          dtype=dtype)

        # if initializing using same seed as non-dist version
        # adjust size of self.weights for halo dimensions
        out_indices = []
        for cur_channel in range(self.nofm):
            self.backend.begin()
            current_index = (cur_channel * self.global_ofmsize +
                             top_left_row_output * self.global_ofmwidth +
                             top_left_col_output)
            for cur_row in range(self.ofmheight):
                self.backend.begin()
                out_indices.extend(
                    range(current_index, current_index + self.ofmwidth))
                current_index += self.global_ofmwidth
                self.backend.end()
            self.backend.end()
        self.weights = self.weights.take(out_indices, axis=0)

        self.normalize_weights(self.weights)
        self.updates = self.backend.empty(self.weights.shape)
        self.learning_rule.allocate_state(self.updates)
        self.prodbuf = self.backend.empty((self.nofm, self.batch_size))
        self.bpropbuf = self.backend.empty((self.fsize, self.batch_size))
        self.updatebuf = self.backend.empty((self.nofm, self.fsize))

    def __init__(self, name, backend, batch_size, pos, learning_rule,
                 nifm, nofm, ifmshape, fshape, stride, weight_init,
                 pretraining, sparsity, tied_weights, prev_names=[]):
        super(
            LocalFilteringLayerDist, self).__init__(name, backend, batch_size,
                                                    pos, learning_rule,
                                                    nifm, nofm, ifmshape,
                                                    fshape, stride,
                                                    prev_names=prev_names)
        self.nout = self.ofmsize * nofm
        self.output = backend.empty((self.nout, batch_size))
        self.weight_init = weight_init
        self.weights = self.weight_init.generate((self.nout, self.fsize),
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
        self.backend.divide(error, inputs.shape[1], out=error)
        self.defilter.bprop(error, self.output)
        self.defilter.update(epoch)
        # Now backward propagate the gradient of the output of the
        # pooling layer.
        error = ((self.sparsity / inputs.shape[1]) *
                 (self.backend.ones(self.pooling.output.shape)))
        self.pooling.bprop(error, self.output)
        deltas = self.defilter.deltas + (
            self.pooling.input.get_bprop_view(self.pooling.deltas))
        self.bprop(deltas, inputs)
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
            self.weights = prev.backend.copy(prev.weights)
        self.updates = prev.backend.empty(self.weights.shape)
        self.prodbuf = prev.backend.empty((prev.fsize, prev.batch_size))
        self.bpropbuf = prev.backend.empty((prev.nofm, prev.batch_size))
        self.updatebuf = prev.backend.empty((prev.nofm, prev.fsize))
        self.deltas = prev.backend.empty((prev.nout, prev.batch_size))
        self.temp = [prev.backend.empty(self.output.shape)]
        self.learning_rule = prev.learning_rule
        self.learning_rule.set_pretrain_mode(True)
        self.backend = prev.backend
        self.rlinks = prev.rlinks
        self.prev = prev

    def fprop(self, inputs):
        self.output.fill(0)
        for dst in range(self.prev.ofmsize):
            self.backend.begin()
            rflinks = self.rlinks[dst]
            # size guide:
            # inputs[:, self.prev.ofmlocs[dst]]: mbs x nout -> mbs x nofm
            # self.weights.take: nofm x ifmsize
            self.backend.dot(self.weights.take(self.prev.ofmlocs[dst],
                                               axis=0).transpose(),
                             inputs[self.prev.ofmlocs[dst]],
                             out=self.prodbuf)
            output_slice = self.output[rflinks]
            self.backend.add(output_slice, self.prodbuf, output_slice)
            self.backend.end()

    def bprop(self, error, inputs):
        for dst in range(self.prev.ofmsize):
            self.backend.begin()
            rflinks = self.rlinks[dst]
            self.backend.dot(self.weights.take(self.prev.ofmlocs[dst],
                                               axis=0),
                             error[rflinks],
                             out=self.bpropbuf)
            self.deltas[self.prev.ofmlocs[dst]] = self.bpropbuf
            delta_slice = error[rflinks]
            self.backend.dot(inputs[self.prev.ofmlocs[dst]],
                             delta_slice.transpose(),
                             out=self.updatebuf)
            self.updates[self.prev.ofmlocs[dst]] = self.updatebuf
            self.backend.end()

    def update(self, epoch):
        self.learning_rule.apply_rule(self.weights, self.updates, epoch)
        self.prev.normalize_weights(self.weights)


class MaxPoolingLayer(LocalLayer):

    """
    Max pooling layer.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride, prev_names=[]):
        super(MaxPoolingLayer, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm, ifmshape,
            fshape, stride, pooling=True, prev_names=prev_names)
        self.maxinds = backend.empty((self.ofmsize, batch_size * nifm),
                                     dtype='i16')
        self.nout = self.nifm * self.ofmsize
        self.output = self.backend.empty((self.nout, batch_size))
        assert fshape[0] * fshape[1] <= 2 ** 15

    def __str__(self):
        temp = self.backend.empty((1, 1))
        return ("MaxPoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t"
                "maxinds: mean=%.05f, min=%.05f, max=%.05f\n\t"
                "output: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__,
                 float(self.backend.mean(self.maxinds, axes=None,
                                         out=temp).asnumpyarray()),
                 float(self.backend.min(self.maxinds, axes=None,
                                        out=temp).asnumpyarray()),
                 float(self.backend.max(self.maxinds, axes=None,
                                        out=temp).asnumpyarray()),
                 float(self.backend.mean(self.output, axes=None,
                                         out=temp).asnumpyarray()),
                 float(self.backend.min(self.output, axes=None,
                                        out=temp).asnumpyarray()),
                 float(self.backend.max(self.output, axes=None,
                                        out=temp).asnumpyarray())))

    def fprop(self, inputs):
        self.backend.fprop_pool(out=self.output, inputs=inputs, op="max",
                                ofmshape=self.ofmshape,
                                ofmsize=self.ofmsize,
                                ofmlocs=self.maxinds,
                                fshape=self.fshape, ifmshape=self.ifmshape,
                                links=self.links, nifm=self.nifm, padding=0,
                                stride=self.stride, fpropbuf=self.outputbuf)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_pool(out=self.deltas, fouts=self.output,
                                    inputs=inputs, deltas=error, op="max",
                                    ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=self.maxinds, fshape=self.fshape,
                                    fpsize=self.fpsize,
                                    ifmshape=self.ifmshape, links=self.links,
                                    nifm=self.nifm, padding=0,
                                    stride=self.stride,
                                    bpropbuf=self.deltasbuf)

    def update(self, epoch):
        pass


class MaxPoolingLayerDist(LocalLayerDist, MaxPoolingLayer):

    """
    Distributed Max pooling layer.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride, prev_names=[]):
        super(MaxPoolingLayerDist, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm, ifmshape,
            fshape, stride, pooling=True, prev_names=prev_names)
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
                 stride, prev_names=[]):
        super(L2PoolingLayer, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True, prev_names=prev_names)
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
        self.backend.fprop_pool(out=self.output, inputs=inputs, op="l2",
                                ofmshape=self.ofmshape,
                                ofmsize=self.ofmsize,
                                ofmlocs=None,
                                fshape=self.fshape, ifmshape=self.ifmshape,
                                links=self.links, nifm=self.nifm, padding=0,
                                stride=self.stride, fpropbuf=self.outputbuf)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_pool(out=self.deltas, fouts=self.output,
                                    inputs=inputs, deltas=error, op="l2",
                                    ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=self.prodbuf, fshape=self.fshape,
                                    fpsize=self.fpsize,
                                    ifmshape=self.ifmshape, links=self.links,
                                    nifm=self.nifm, padding=0,
                                    stride=self.stride,
                                    bpropbuf=self.deltasbuf)

    def update(self, epoch):
        pass


class L2PoolingLayerDist(LocalLayerDist, L2PoolingLayer):

    """
    Distributed L2 pooling layer. Each receptive field is pooled to obtain its
    L2 norm as output.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride, prev_names=[]):
        super(L2PoolingLayerDist, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True, prev_names=prev_names)
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
                 stride, prev_names=[]):
        super(AveragePoolingLayer, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True, prev_names=prev_names)
        self.nout = nifm * self.ofmsize
        self.output = self.backend.empty((self.nout, batch_size))

    def __str__(self):
        return ("AveragePoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        self.backend.fprop_pool(out=self.output, inputs=inputs, op="avg",
                                ofmshape=self.ofmshape,
                                ofmsize=self.ofmsize,
                                ofmlocs=None,
                                fshape=self.fshape, ifmshape=self.ifmshape,
                                links=self.links, nifm=self.nifm, padding=0,
                                stride=self.stride, fpropbuf=self.outputbuf)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_pool(out=self.deltas, fouts=self.output,
                                    inputs=inputs, deltas=error, op="avg",
                                    ofmshape=self.ofmshape,
                                    ofmsize=self.ofmsize,
                                    ofmlocs=None,
                                    fshape=self.fshape,
                                    fpsize=self.fpsize,
                                    ifmshape=self.ifmshape,
                                    links=self.links, nifm=self.nifm,
                                    padding=0, stride=self.stride,
                                    bpropbuf=self.deltasbuf)

    def update(self, epoch):
        pass


class AveragePoolingLayerDist(LocalLayerDist, AveragePoolingLayer):

    """
    Distributed Average pooling layer.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride, prev_names=[]):
        super(AveragePoolingLayerDist, self).__init__(
            name, backend, batch_size, pos, 0.0, nifm, nifm,
            ifmshape, fshape, stride, pooling=True, prev_names=prev_names)
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
            self.backend.begin()
            rflinks = self.rlinks[dst]
            self.backend.dot(self.weights, inputs.take(rflinks, axis=0),
                             out=self.prodbuf)
            self.output[self.ofmlocs[dst]] = self.prodbuf
            self.backend.end()


class LCNLayer(YAMLable):

    """
    Local contrast normalization.
    """

    def __init__(self, name, backend, batch_size, pos, nifm, ifmshape, fshape,
                 stride, prev_names=[]):
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
        self.prev_names = prev_names

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
                self.backend.begin()
                self.bprop_filters[fm] = self.backend.copy(self.filters)
                rfilter = self.bprop_filters[fm].reshape(
                    (nifm, self.fheight, self.fwidth))
                fm_filt = rfilter[fm, self.fheight / 2, self.fwidth / 2]
                self.backend.subtract(fm_filt, 1.0, fm_filt)
                self.backend.end()

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
        assert self.subout[self.meanfm.asnumpyarray() == 0.0].asnumpyarray(
            ).sum() == 0.0
        self.meanfm[self.meanfm.asnumpyarray() == 0.0] = 1.0
        self.backend.divide(self.rsubout, self.rmeanfm, out=self.routput)

    def fprop(self, inputs):
        self.exinputs.fill(0)
        self.fprop_sub_normalize(inputs)
        self.fprop_div_normalize()

    def reshape_error(self):
        # discards zero padding around the delta matrix
        self.deltas = self.copy_from_inset(self.rexerror, self.start_row,
                                           self.start_col)
        self.deltas = self.deltas.reshape((self.nin, self.batch_size))

    def bprop_sub_normalize(self, error, inputs):
        self.exerror.fill(0)
        for fm in range(self.nifm):
            self.backend.begin()
            for dst in range(self.conv.ofmsize):
                self.backend.begin()
                rflinks = self.conv.rlinks[dst]
                loc = (self.conv.ofmlocs[dst].asnumpyarray().squeeze() +
                       self.conv.ofmsize * fm)
                filt = self.bprop_filters[fm]
                self.backend.multiply(error[loc].transpose(), filt.transpose(),
                                      out=self.prodbuf)
                exerror_slice = self.exerror[rflinks]
                self.backend.subtract(exerror_slice, self.prodbuf,
                                      exerror_slice)
                self.backend.end()
            self.backend.end()
        self.reshape_error()

    def bprop_div_normalize(self, error, inputs):
        self.exerror.fill(0)
        self.backend.cube(self.output, out=self.diverror)
        self.subtemp[:] = self.subout
        assert self.diverror[self.subout.asnumpyarray() == 0].asnumpyarray(
            ).sum() == 0.0
        self.subout[self.subout.asnumpyarray() == 0] = 1.0
        self.backend.square(self.subout, out=self.sqtemp)
        # this is for the non-padded, non-halo matrix only
        self.backend.divide(self.diverror, self.sqtemp, out=self.diverror)

        for fm in range(self.nifm):
            self.backend.begin()
            for dst in range(self.conv.ofmsize):
                self.backend.begin()
                # self.conv.ofmlocs is over 1 fm only
                loc = (self.conv.ofmlocs[dst].asnumpyarray().squeeze() +
                       self.conv.ofmsize * fm)
                divout = self.output.take(loc, axis=0).transpose()
                subout = self.subout.take(loc, axis=0).transpose()
                assert divout[subout.asnumpyarray() == 0].asnumpyarray(
                    ).sum() == 0
                subout[subout.asnumpyarray() == 0.0] = 1.0
                self.backend.divide(divout, subout, out=divout)

                rflinks = self.conv.rlinks[dst]
                self.copy_to_inset(self.rexinputs, self.rsubtemp,
                                   self.start_row, self.start_col)
                rrexinputs = self.rexinputs.reshape(
                    (self.nifm * self.exifmsize, self.batch_size))
                frame = rrexinputs.take(rflinks, axis=0)
                self.backend.multiply(frame, self.filters.transpose(),
                                      out=frame)
                self.backend.multiply(frame, self.diverror[loc].transpose(),
                                      out=frame)
                rframe = frame.reshape((self.nifm, self.fheight, self.fwidth,
                                        self.batch_size))
                # this is working on the g2/y2 term
                rframe_slice = rframe[fm:(fm + 1), self.fheight / 2,
                                      self.fwidth / 2]
                self.backend.subtract(rframe_slice, divout, rframe_slice)
                self.backend.multiply(error[loc].transpose(), frame, out=frame)
                exerror_slice = self.exerror[rflinks]
                self.backend.subtract(exerror_slice, frame, exerror_slice)
                self.backend.end()
            self.backend.end()
        self.reshape_error()

    def bprop(self, error, inputs):
        if self.pos > 0:
            # note: have to account for halos + padding after each step
            self.bprop_div_normalize(error, inputs)
            self.bprop_sub_normalize(self.deltas, inputs)

    def bprop_fast(self, error, inputs):
        """
        An incorrect, but much faster version of backprop.
        """
        if self.pos > 0:
            self.deltas[:] = error

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
                self.backend.begin()
                self.bprop_filters[fm] = self.backend.copy(self.filters)
                rfilter = self.bprop_filters[fm].reshape(
                    (self.nifm, self.fheight, self.fwidth))
                rfilter[fm, self.fheight / 2, self.fwidth / 2] -= 1.0
                self.backend.end()

    def copy_to_inset(self, canvas, inset, start_row, start_col):
        canvas[:, start_row:start_row + inset.shape[1],
               start_col:start_col + inset.shape[2]] = inset

    def copy_from_inset(self, canvas, start_row, start_col):
        return canvas[:, start_row:start_row + self.ifmheight,
                      start_col:start_col + self.ifmwidth]

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
                               self.rexinputs.shape[2] - self.start_col2)],
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
        assert self.subout[self.meanfm.asnumpyarray() == 0.0].sum() == 0.0
        self.meanfm[self.meanfm.asnumpyarray() == 0.0] = 1.0
        self.backend.divide(
            self.input.get_local_acts().reshape(
                self.routput.shape), self.rmeanfm, out=self.routput)

    def fprop(self, inputs_):
        self.exinputs.fill(0)
        inputs = self.input.get_fprop_view(inputs_)
        self.fprop_sub_normalize(inputs)
        # distributed version
        self.input.make_fprop_view(self.subout)
        self.fprop_div_normalize()

    def bprop_div_normalize(self, error, inputs):
        self.exerror.fill(0)
        self.backend.cube(self.output, out=self.diverror)

        self.subout = self.input.get_local_acts()
        self.subtemp2[:] = self.input.local_array.chunk

        self.subtemp[:] = self.subout
        assert self.diverror[self.subout.asnumpyarray() == 0].sum() == 0.0
        self.subout[self.subout.asnumpyarray() == 0] = 1.0
        self.backend.square(self.subout, out=self.sqtemp)
        # this is for the non-padded, non-halo matrix only
        self.backend.divide(self.diverror, self.sqtemp, out=self.diverror)

        for fm in range(self.nifm):
            self.backend.begin()
            for dst in range(self.conv.ofmsize):
                self.backend.begin()
                # self.conv.ofmlocs is over 1 fm only
                loc = (self.conv.ofmlocs[dst].asnumpyarray() +
                       self.conv.ofmsize * fm)
                divout = self.output.take(loc, axis=0)
                subout = self.subout.take(loc, axis=0)
                assert divout[subout.asnumpyarray() == 0].sum() == 0
                subout[subout.asnumpyarray() == 0.0] = 1.0
                self.backend.divide(divout, subout, out=divout)

                rflinks = self.conv.rlinks[dst]
                self.copy_to_inset(self.rexinputs, self.rsubtemp2,
                                   self.start_row, self.start_col)
                rrexinputs = self.rexinputs.reshape(
                    (self.nifm * self.exifmsize, self.batch_size))
                frame = rrexinputs.take(rflinks, axis=0)
                self.backend.multiply(frame, self.filters.transpose(),
                                      out=frame)
                self.backend.multiply(frame, self.diverror[loc], out=frame)
                rframe = frame.reshape((self.nifm,
                                        self.fheight, self.fwidth,
                                        self.batch_size))
                # this is working on the g2/y2 term
                rframe[fm:(fm + 1),
                       self.fheight / 2, self.fwidth / 2] -= divout
                self.backend.multiply(error[loc].repeat(self.fsize, axis=0),
                                      frame, out=frame)
                self.exerror[rflinks] -= frame
                self.backend.end()
            self.backend.end()
        self.reshape_error()

    def bprop(self, error, inputs):
        if self.pos > 0:
            # note: have to account for halos + padding after each step
            self.bprop_div_normalize(error, inputs)

            self.bprop_sub_normalize(self.input.get_bprop_view(self.deltas),
                                     inputs)

            self.deltas = (self.input.get_bprop_view(self.deltas))


class CrossMapPoolingLayer(YAMLable):

    """
    Pool input feature maps by computing a weighted sum of
    corresponding spatial locations across maps. This is
    equivalent to a 1x1 convolution.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rule,
                 nifm, nofm, ifmshape, weight_init, activation=None,
                 prev_names=[]):
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
        self.prev_names = prev_names
        self.ofmshape = self.ifmshape
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ifmsize
        self.nin = nifm * self.ifmsize
        self.nout = nofm * self.ifmsize
        if pos > 0:
            self.deltas = backend.empty((self.nin, batch_size))

        self.weights = weight_init.generate((nifm, nofm))
        assert (self.weights.asnumpyarray() < 0).sum() == 0
        self.updates = backend.empty(self.weights.shape)
        self.output = backend.empty((self.nout, batch_size))
        self.updatebuf = backend.empty((1, 1))
        self.learning_rule.allocate_state([self.updates])
        if activation is not None:
            self.pre_act = backend.empty((self.nout, batch_size))
        else:
            self.pre_act = self.output

    def fprop(self, inputs):
        self.backend.fprop_cmpool(out=self.pre_act, inputs=inputs,
                                  weights=self.weights, ifmshape=self.ifmshape,
                                  ifmsize=self.ifmsize)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs):
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)
        if self.pos > 0:
            self.backend.bprop_cmpool(out=self.deltas, weights=self.weights,
                                      deltas=error, ifmshape=self.ifmshape,
                                      ifmsize=self.ifmsize)
        self.backend.update_cmpool(out=self.updates, inputs=inputs,
                                   deltas=error, ifmshape=self.ifmshape,
                                   ifmsize=self.ifmsize,
                                   updatebuf=self.updatebuf)

    def update(self, epoch):
        self.learning_rule.apply_rule([self.weights], [self.updates], epoch)

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
                 nifm, ksize, alpha, beta, prev_names=[]):

        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nifm = nifm
        self.nin = self.ifmsize * self.nifm
        self.nout = self.nin

        self.name = name
        self.backend = backend
        self.ifmshape = ifmshape
        self.batch_size = batch_size
        self.pos = pos
        self.prev_names = prev_names
        self.ksize = ksize
        self.alpha = alpha * 1.0 / ksize
        self.beta = beta

        self.output = self.backend.empty((self.nout, self.batch_size))
        if self.pos > 0:
            self.deltas = self.backend.empty((self.nin, self.batch_size))
            self.tempbuf = self.backend.empty((ifmshape[0], ifmshape[1],
                                               batch_size))

    def fprop(self, inputs):
        self.backend.fprop_cmrnorm(out=self.output, inputs=inputs,
                                   ifmshape=self.ifmshape, nifm=self.nifm,
                                   ksize=self.ksize, alpha=self.alpha,
                                   beta=self.beta)

    def bprop(self, error, inputs):
        if self.pos > 0:
            self.backend.bprop_cmrnorm(out=self.deltas, fouts=self.output,
                                       inputs=inputs, deltas=error,
                                       ifmshape=self.ifmshape, nifm=self.nifm,
                                       ksize=self.ksize, alpha=self.alpha,
                                       beta=self.beta, bpropbuf=self.tempbuf)

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        pass
