import logging
import numpy as np
from neon.models import learning_rule as lr
from neon.models.layer2 import Layer, CostLayer
from neon.util.compat import range
from neon.util.param import req_param, opt_param
from ipdb import set_trace as trace

logger = logging.getLogger(__name__)


class RecurrentLayer(Layer):
    """
    This is a dummy layer to prevent inheritance from outside the rlayer2 file
    Long inheritance chains like RecurrentHiddenLayer<-RecurrentOutputLayer<-
    WeightLayer<-Layer are extremely confusing for me, but I'm not sure if
    this is a clean way to do it either.
    Inheriting Layer seems to be mandatory for the yaml stuff...
    """

    def allocate_output_bufs(self):
        make_zbuf = self.backend.zeros
        # basically is it possible to set these opt params somewhere else?
        opt_param(self, ['out_shape'], (self.nout, self.batch_size))

        self.output = make_zbuf(self.out_shape, self.output_dtype)

        if self.activation is not None:
            self.pre_act = make_zbuf(self.out_shape, self.pre_act_dtype)
        else:
            self.pre_act = self.output

        """ create deltas buffer no matter what position relative to the data
        layer we are. In the RNN even the first layer needs deltas."""
        self.deltas = make_zbuf(self.delta_shape, self.deltas_dtype)

    # c&p from WeightLayer
    def set_previous_layer(self, pl):
        if pl.is_local:
            if self.is_local:
                self.ifmshape = pl.ofmshape
                self.nifm = pl.nofm
            self.nin = pl.nofm * np.prod(pl.ofmshape)
        else:
            if self.is_local:
                if not hasattr(self, 'ifmshape'):
                    sqdim = np.int(np.sqrt(pl.nout))
                    self.ifmshape = (sqdim, sqdim)
                self.nifm = 1
            self.nin = pl.nout
        self.prev_layer = pl

    # No longer sure about how smart it is to c&p all this WeightLayer stuff
    def allocate_param_bufs(self):
        make_ebuf = self.backend.empty
        self.weights = self.weight_init.generate(self.weight_shape,
                                                 self.weight_dtype)
        self.weight_updates = make_ebuf(self.weight_shape, self.updates_dtype)

        self.use_biases = 'bias_init' in self.weight_init.__dict__
        opt_param(self, ['brule_init'], None)
        if self.use_biases is True:
            self.biases = make_ebuf(self.bias_shape, self.weight_dtype)
            self.biases.fill(self.weight_init.bias_init)
            self.bias_updates = make_ebuf(self.bias_shape, self.updates_dtype)
            self.params = [self.weights, self.biases]
            self.updates = [self.weight_updates, self.bias_updates]
        else:
            self.params = [self.weights]
            self.updates = [self.weight_updates]

        if self.accumulate:
            self.utemp = map(lambda x: make_ebuf(x.shape, self.updates_dtype),
                             self.updates)
        for upm in self.updates:
            upm.fill(0.0)
        self.learning_rule = self.init_learning_rule(self.lrule_init)
        self.bias_rule = None
        if self.brule_init is not None and self.use_biases:
            self.bias_rule = self.init_learning_rule(self.brule_init)
            self.bias_rule.allocate_state([self.bias_updates])
            self.learning_rule.allocate_state([self.weight_updates])
        else:
            self.learning_rule.allocate_state(self.updates)

    # more c&p from WeightLayer, may want to revert this!
    def init_learning_rule(self, lrule_init):
        lrname = self.name + '_lr'
        if lrule_init['type'] == 'gradient_descent':
            return lr.GradientDescent(name=lrname,
                                      lr_params=lrule_init['lr_params'])
        elif lrule_init['type'] == 'gradient_descent_pretrain':
            return lr.GradientDescentPretrain(
                name=lrname, lr_params=lrule_init['lr_params'])
        elif lrule_init['type'] == 'gradient_descent_momentum':
            return lr.GradientDescentMomentum(
                name=lrname, lr_params=lrule_init['lr_params'])
        elif lrule_init['type'] == 'gradient_descent_momentum_weight_decay':
            return lr.GradientDescentMomentumWeightDecay(
                name=lrname, lr_params=lrule_init['lr_params'])
        elif lrule_init['type'] == 'adadelta':
            return lr.AdaDelta(name=lrname, lr_params=lrule_init['lr_params'])
        else:
            raise AttributeError("invalid learning rule params specified")


class RecurrentCostLayer(CostLayer):
    '''Not sure if this is needed, but the bprop fails here? '''
    def __init__(self, **kwargs):
        self.is_cost = True
        self.nout = 1
        super(RecurrentCostLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        super(RecurrentCostLayer, self).initialize(kwargs)
        req_param(self, ['cost', 'ref_layer'])
        opt_param(self, ['ref_label'], 'targets')
        self.targets = None
        self.cost.olayer = self.prev_layer
        self.cost.initialize(kwargs)
        self.deltas = self.cost.get_deltabuf()

    def __str__(self):
        return ("Layer {lyr_nm}: {nin} nodes, {cost_nm} cost_fn, "
                "utilizing {be_nm} backend\n\t".format
                (lyr_nm=self.name, nin=self.nin,
                 cost_nm=self.cost.__class__.__name__,
                 be_nm=self.backend.__class__.__name__))

    def fprop(self, inputs):
        pass

    def bprop(self, error, tau):
        # Since self.deltas already pointing to destination of act gradient
        # we just have to scale by mini-batch size
        '''for recurrent networks, the targets are a list of unrolls'''
        if self.ref_layer is not None:
            self.targets = getattr(self.ref_layer, self.ref_label)
        # if self.ref_label != 'targets':
        #     print self.targets.shape
        '''this stuff was done in rnn.fit(), only applied to last output
        so I'm not sure this statement needs to be conditional on tau...'''
        self.cost.apply_derivative(self.targets[tau])
        self.backend.divide(self.deltas, self.batch_size, out=self.deltas)

    def get_cost(self):
        #race() # not sure if it's cool to just use the last one?
        result = self.cost.apply_function(self.targets[-1])
        return self.backend.divide(result, self.batch_size, result)


class RecurrentOutputLayer(RecurrentLayer):

    """
    Derived from Layer. pre_act becomes pre_act_list, output becomes
    output_list, which are indexed by [tau], the unrolling step.
    """
    def initialize(self, kwargs):
        req_param(self, ['nout', 'nin', 'unrolls', 'activation'])
        super(RecurrentOutputLayer, self).initialize(kwargs)
        self.weight_shape = (self.nout, self.nin)
        self.bias_shape = (self.nout, 1)

        opt_param(self, ['delta_shape'], (self.nin, self.batch_size))  # moved
        self.allocate_output_bufs()
        self.allocate_param_bufs()

    def allocate_output_bufs(self):
        super(RecurrentOutputLayer, self).allocate_output_bufs()
        # super allocate will set the correct sizes for pre_act, output, berr
        make_zbuf = self.backend.zeros

        self.pre_act_list = [self.pre_act] + \
                            [make_zbuf(self.out_shape, self.pre_act_dtype)
                             for k in range(1, self.unrolls)]
        self.output_list = [self.output] + \
                           [make_zbuf(self.out_shape, self.output_dtype)
                            for k in range(1, self.unrolls)]
        self.temp_out = make_zbuf(self.weight_shape, self.weight_dtype)

    def fprop(self, inputs, tau):
        self.backend.fprop_fc(self.pre_act_list[tau], inputs, self.weights)
        self.activation.apply_both(self.backend,
                                   self.pre_act_list[tau],
                                   self.output_list[tau])

    def bprop(self, error, tau, numgrad=False):
        inputs = self.prev_layer.output_list[tau - 1]
        if self.skip_act is False:
            self.backend.multiply(error, self.pre_act_list[tau - 1], error)

        self.backend.bprop_fc(self.deltas, self.weights, error)
        self.backend.update_fc(out=self.temp_out, inputs=inputs, deltas=error)

        if numgrad == "output":
            self.grad_log(numgrad, self.temp_out[12, 56])

        self.backend.add(self.weight_updates, self.temp_out,
                         self.weight_updates)

    def grad_log(self, ng, val):
        logger.info("%s.bprop inc %s %f", self.__class__.__name__, ng, val)


class RecurrentHiddenLayer(RecurrentLayer):

    """
    Derived from Layer. In addition to the lists[tau] outlined for
    RecurrentOutputLayer, the fprop is getting input from two weight matrices,
    one connected to the input and one connected to the previous hidden state.
    """
    def initialize(self, kwargs):
        req_param(self, ['weight_init_rec'])
        self.weight_rec_shape = (self.nout, self.nout)
        super(RecurrentHiddenLayer, self).initialize(kwargs)

        # c&p from ROL
        self.weight_shape = (self.nout, self.nin)
        self.bias_shape = (self.nout, 1)

        # Solution: Set delta_shape to nout since that's the correct size for
        # the recurrent deltas
        opt_param(self, ['delta_shape'], (self.nout, self.batch_size)) # moved
        self.allocate_output_bufs()
        self.allocate_param_bufs()

    def allocate_output_bufs(self):
        super(RecurrentHiddenLayer, self).allocate_output_bufs()

        # c&p from ROL
        make_zbuf = self.backend.zeros
        self.pre_act_list = [self.pre_act] + \
                            [make_zbuf(self.out_shape, self.pre_act_dtype)
                             for k in range(1, self.unrolls)]
        self.output_list = [self.output] + \
                            [make_zbuf(self.out_shape, self.output_dtype)
                             for k in range(1, self.unrolls)]
        self.temp_out = make_zbuf(self.weight_shape, self.weight_dtype)

        # these buffers are specific to RHL it seems
        self.temp_in = self.temp_out
        self.temp_rec = self.backend.zeros(self.weight_rec_shape)
        self.z = [self.backend.zeros(self.out_shape) for k in range(2)]

    def allocate_param_bufs(self):
        super(RecurrentHiddenLayer, self).allocate_param_bufs()
        self.weights_rec = self.weight_init_rec.generate(
                                            self.weight_rec_shape,
                                            self.weight_dtype)

        self.updates_rec = self.backend.empty(self.weight_rec_shape,
                                              self.updates_dtype)

        self.params.append(self.weights_rec)
        self.updates.append(self.updates_rec)
        # Not ideal, since we just allocated this in the parent function, but
        # we can change the calling order later
        self.learning_rule.allocate_state(self.updates)

        # nothing to c&p since ROL does not have this and passes through to RL

    def fprop(self, y, inputs, tau, cell=None):
        self.backend.fprop_fc(self.z[0], y, self.weights_rec)
        self.backend.fprop_fc(self.z[1], inputs, self.weights)
        self.backend.add(self.z[0], self.z[1], self.pre_act_list[tau])
        self.activation.apply_both(self.backend,
                                   self.pre_act_list[tau],
                                   self.output_list[tau])

    def bprop(self, error, error_c, tau, t, numgrad=False):
        """
        This function has been refactored:
        [done] remove duplicate code
        [done] remove the loop altogether.
        [todo] If the if statement can't be supported, revert to duplicated
               code
        Not sure why tau is passed but not used. Not that this is called for
        decrementing t.
        """
        # Is this equivalent to the old way of doing
        # inputs[t*self.nin:(t+1)*self.nin, :]
        # with the new datalayer format?
        if self.prev_layer.is_data:
            inputs = self.prev_layer.output[t]
        else:
            inputs = self.prev_layer.output_list[t]

        if self.skip_act is False:
            self.backend.multiply(error, self.pre_act_list[t], out=error)

        # input weight update (apply curr. delta)

        self.backend.update_fc(out=self.temp_in,
                               inputs=inputs,
                               deltas=error)
        self.backend.add(self.weight_updates, self.temp_in,
                         self.weight_updates)

        if (t > 0):
            # recurrent weight update (apply prev. delta)
            self.backend.update_fc(out=self.temp_rec,
                                   inputs=self.output_list[t - 1],  # avoid t=0
                                   deltas=error)
            self.backend.add(self.updates_rec, self.temp_rec, self.updates_rec)

            # **** ASK URS ***
            # Why only at t > 0 vs. t==0? why not weights vs weights_rec
            self.backend.bprop_fc(out=self.deltas,  # output for next iteration
                                  weights=self.weights_rec,
                                  deltas=error)
        if numgrad == "input":
            self.grad_log(numgrad, self.temp_in[12, 110])
        if numgrad == "rec":
            self.grad_log(numgrad, self.temp_rec[12, 63])


class RecurrentLSTMLayer(RecurrentLayer):

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
            setattr(self, a + '_t',
                    [be.zeros(net_sze) for k in range(unrolls)])
            setattr(self, 'net_' + a,
                    [be.zeros(net_sze) for k in range(unrolls)])

        for a in ['i', 'f', 'o', 'c']:
            setattr(self, 'W' + a + 'x',
                    be.gen_weights((nout, nin), weight_init_rec, weight_dtype))
            setattr(self, 'W' + a + 'h', be.gen_weights((nout, nout),
                                                        weight_init_rec,
                                                        weight_dtype))
            setattr(self, 'b_' + a, be.zeros((nout, 1)))
            setattr(self, 'W' + a + 'x_updates', be.zeros((nout, nin)))
            setattr(self, 'W' + a + 'h_updates', be.zeros((nout, nout)))
            setattr(self, 'b_' + a + '_updates', be.zeros((nout, 1)))

        # pre-allocate for d{i,f,o,c}_dh1
        self.d_dh1 = {gateid: be.zeros(net_sze) for
                      gateid in ['i', 'f', 'o', 'c']}
        self.dc_d_dh1 = {gateid: be.zeros(net_sze) for
                         gateid in ['i', 'f', 'c']}
        self.errs = {hcval: be.zeros(net_sze) for
                     hcval in ['hh', 'hc', 'ch', 'cc']}
        self.gatedic = {}
        self.gatedic_u = {}

        for a in ['i', 'f', 'o', 'c']:
            gateid = 'g' if a is 'c' else a
            self.gatedic[a] = [getattr(self, 'W' + a + 'x'),
                               getattr(self, 'W' + a + 'h'),
                               getattr(self, 'b_' + a),
                               getattr(self, 'net_' + gateid),
                               getattr(self, gateid + '_t')]
            self.gatedic_u[a] = [getattr(self, 'W' + a + 'x_updates'),
                                 getattr(self, 'W' + a + 'h_updates'),
                                 getattr(self, 'b_' + a + '_updates')]

        # If this isn't initialized correctly, get NaNs pretty quickly.
        be.add(self.b_i, 1, self.b_i)  # sigmoid(1) opens the gate
        # +5 following clockwork RNN paper "to encourage long term memory"
        be.add(self.b_f, -1, self.b_f)  # sigmoid(-1) closes gate.
        be.add(self.b_o, 1, self.b_o)   # sigmoid(1) open

        # and for higher up entities in the LSTM cell.
        self.c_t = [be.zeros(net_sze) for k in range(unrolls)]
        self.c_phi = [be.zeros(net_sze) for k in range(unrolls)]
        self.c_phip = [be.zeros(net_sze) for k in range(unrolls)]
        self.output_list = [be.zeros(net_sze) for k in range(unrolls)]

        # pre-allocate preactivation buffers
        self.temp_x = [be.zeros(net_sze) for k in range(unrolls)]
        self.temp_h = [be.zeros(net_sze) for k in range(unrolls)]

        # pre-allocate derivative buffers
        self.dh_dwx_buf = be.zeros((nout, nin))
        self.dh_dwh_buf = be.zeros((nout, nout))

        self.delta_buf = be.zeros(net_sze)
        self.bsum_buf = be.zeros((nout, 1))

        # This quantity seems to be computed repeatedly
        # error_h * self.o_t[tau] * self.c_phip[tau]
        self.eh_ot_cphip = be.zeros(net_sze)

        self.param_names = ['input', 'forget', 'output', 'cell']
        self.params = [self.Wix, self.Wfx, self.Wox, self.Wcx, self.Wih,
                       self.Wfh, self.Woh, self.Wch, self.b_i, self.b_f,
                       self.b_o, self.b_c]
        self.updates = [self.Wix_updates, self.Wfx_updates, self.Wox_updates,
                        self.Wcx_updates, self.Wih_updates, self.Wfh_updates,
                        self.Woh_updates, self.Wch_updates, self.b_i_updates,
                        self.b_f_updates, self.b_o_updates, self.b_c_updates]

        self.learning_rule.allocate_state(self.updates)
        for upm in self.updates:
            upm.fill(0.0)
        self.deltas = be.zeros((nout, batch_size))  # hidden bprop error
        self.celtas = be.zeros((nout, batch_size))  # cell bprop error

        self.temp_t = 0

    def list_product(self, target, plist):
        """
        Computes the product of the items in list and puts it into target
        """
        target.fill(1.0)
        reduce(lambda x, y: self.backend.multiply(x, y, x), [target] + plist)

    def list_sum(self, target, slist):
        """
        Computes the sum of the items in slist and puts it into target
        """
        target.fill(0.0)
        reduce(lambda x, y: self.backend.add(x, y, x), [target] + slist)

    def cell_bprop(self, delta_buf, xx, yy, tau, gate, dh1_out):
        be = self.backend
        [wx, wh, b] = self.gatedic[gate][:3]
        [wxu, whu, bu] = self.gatedic_u[gate]

        be.bprop_fc(out=dh1_out, weights=wh, deltas=delta_buf)
        be.update_fc(out=self.dh_dwx_buf, inputs=xx, deltas=delta_buf)
        be.update_fc(out=self.dh_dwh_buf, inputs=yy, deltas=delta_buf)
        if (tau > 0):
            # was h only, but Urs changed this to skip the last x as well
            be.add(wxu, self.dh_dwx_buf, wxu)
            be.add(whu, self.dh_dwh_buf, whu)
        be.sum(delta_buf, 1, self.bsum_buf)
        be.add(bu, self.bsum_buf, bu)

    def cell_fprop(self, xx, yy, tau, gate, actfunc):
        be = self.backend
        [wx, wh, b, netl, tl] = self.gatedic[gate]

        be.fprop_fc(self.temp_x[tau], xx, wx)
        be.fprop_fc(self.temp_h[tau], yy, wh)
        be.add(self.temp_x[tau], self.temp_h[tau], netl[tau])
        be.add(netl[tau], b, netl[tau])
        actfunc.apply_both(be, netl[tau], tl[tau])

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
        self.activation is tanh
        self.gate_activation is logistic
        """
        be = self.backend  # shorthand

        # input gate
        self.cell_fprop(inputs, y, tau, 'i', self.gate_activation)
        # # forget gate
        self.cell_fprop(inputs, y, tau, 'f', self.gate_activation)
        # # output gate
        self.cell_fprop(inputs, y, tau, 'o', self.gate_activation)
        # # classic RNN cell
        self.cell_fprop(inputs, y, tau, 'c', self.activation)

        # combine the parts and compute output.
        # c_phip = c_t = f_t * cell + i_t * g_t
        be.multiply(self.f_t[tau], cell, self.c_t[tau])
        be.multiply(self.i_t[tau], self.g_t[tau], self.c_phip[tau])
        be.add(self.c_t[tau], self.c_phip[tau], self.c_t[tau])
        # Hack to avoid creating a new copy for c_phip, just want assign vals
        be.add(self.c_t[tau], 0.0, self.c_phip[tau])

        self.activation.apply_both(be, self.c_phip[tau], self.c_phi[tau])
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
        cur_input = inputs[tau*self.nin:(tau+1)*self.nin, :]
        cur_output = self.output_list[tau - 1]

        numtemp = {}
        for ifoc in ['i', 'f', 'o', 'c']:
            for hx in ['h', 'x']:
                numtemp[ifoc+hx] = np.zeros((2, 1), dtype=np.float32)

        """--------------------------
        PART 1: original dh2/dh1 terms
        --------------------------"""
        # Precalculate error_h * self.o_t[tau] * self.c_phip[tau]
        self.list_product(self.eh_ot_cphip,
                          [error_h, self.o_t[tau], self.c_phip[tau]])

        # a. Input gate
        # self.delta_buf = error_h * self.o_t[tau] * self.c_phip[tau] \
        #                  * self.g_t[tau] * self.net_i[tau]
        # b. forget gate
        # self.delta_buf = error_h * self.o_t[tau] * self.c_phip[tau] \
        #                  * self.c_t[tau-1] * self.net_f[tau]
        # c. output gate
        # self.delta_buf = error_h * self.c_phi[tau] * self.net_o[tau]
        #
        # d. cell
        # self.delta_buf = error_h * self.o_t[tau] * self.c_phip[tau]
        #                  * self.i_t[tau] * self.net_g[tau]

        deltargs = {'i': [self.eh_ot_cphip, self.g_t[tau], self.net_i[tau]],
                    'f': [self.eh_ot_cphip, self.c_t[tau-1], self.net_f[tau]],
                    'o': [error_h, self.c_phi[tau], self.net_o[tau]],
                    'c': [self.eh_ot_cphip, self.i_t[tau], self.net_g[tau]]}

        for ifoc in ['i', 'f', 'o', 'c']:
            self.list_product(self.delta_buf, deltargs[ifoc])
            self.cell_bprop(self.delta_buf, cur_input, cur_output, tau,
                            ifoc, self.d_dh1[ifoc])
            numtemp[ifoc+'h'][0] = self.dh_dwh_buf[12, 55].asnumpyarray()
            numtemp[ifoc+'x'][0] = self.dh_dwx_buf[12, 110].asnumpyarray()

        # e. collect terms
        self.list_sum(self.errs['hh'], self.d_dh1.values())

        """ --------------------------
        PART 2: New dc2/dc1 dc2/dh1 and dh2/dc1 terms
        ---------------------------"""
        # a. Input gate
        # self.delta_buf = error_c * self.g_t[tau] * self.net_i[tau]
        # b. Forget gate
        # self.delta_buf = error_c * self.c_t[tau-1] * self.net_f[tau]
        # c. cell
        # self.delta_buf = error_c * self.i_t[tau] * self.net_g[tau]
        deltargs = {'i': [error_c, self.g_t[tau], self.net_i[tau]],
                    'f': [error_c, self.c_t[tau-1], self.net_f[tau]],
                    'c': [error_c, self.i_t[tau], self.net_g[tau]]}

        for ifc in ['i', 'f', 'c']:
            self.list_product(self.delta_buf, deltargs[ifc])
            self.cell_bprop(self.delta_buf, cur_input, cur_output, tau,
                            ifc, self.dc_d_dh1[ifc])
            numtemp[ifc+'h'][1] = self.dh_dwh_buf[12, 55].asnumpyarray()
            numtemp[ifc+'x'][1] = self.dh_dwx_buf[12, 110].asnumpyarray()

        # errs['ch'] = sum of dc_d{i,f,g}_dh1 terms
        # errs['hc'] = error_h * self.o_t * self.c_phip * self.f_t @ tau
        # errs['cc'] = error_c * self.f_t[tau]
        self.list_sum(self.errs['ch'], self.dc_d_dh1.values())
        self.list_product(self.errs['hc'], [self.eh_ot_cphip, self.f_t[tau]])
        be.multiply(error_c, self.f_t[tau], self.errs['cc'])

        # wrap up:
        be.add(self.errs['hh'], self.errs['ch'], self.deltas)
        be.add(self.errs['cc'], self.errs['hc'], self.celtas)

        if numgrad is not None and numgrad.startswith("lstm"):
            ifoc_hx = numgrad[5:7]
            logger.info("LSTM.bprop: analytic dh_dw%s[%d]= %e + %e = %e",
                        ifoc_hx, tau, numtemp[ifoc_hx][0], numtemp[ifoc_hx][1],
                        numtemp[ifoc_hx][0] + numtemp[ifoc_hx][1])
