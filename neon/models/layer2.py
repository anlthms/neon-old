# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
import numpy as np
import math as mt
from neon.backends.cpu import CPU
from neon.models import learning_rule as lr
from neon.util.compat import range
from neon.util.param import req_param, opt_param
from neon.util.persist import YAMLable

logger = logging.getLogger(__name__)


class Layer(YAMLable):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        req_param(self, ['name'])

        opt_param(self, ['pre_act_dtype', 'output_dtype', 'berror_dtype'])
        opt_param(self, ['weight_dtype', 'updates_dtype'])
        opt_param(self, ['prev_layer', 'activation'])

        opt_param(self, ['is_local', 'is_data', 'is_cost'], False)
        opt_param(self, ['skip_act'], False)
        opt_param(self, ['prev_names'], [])

    def set_previous_layer(self, pl):
        if pl.is_local:
            if self.is_local:
                self.ifmshape = pl.ofmshape
                self.nifm = pl.nofm
            self.nin = pl.nofm * pl.ofmshape[0] * pl.ofmshape[1]
        else:
            if self.is_local:
                if not hasattr(self, 'ifmshape'):
                    sqdim = np.int(np.sqrt(pl.nout))
                    self.ifmshape = (sqdim, sqdim)
                self.nifm = 1
            self.nin = pl.nout
        self.prev_layer = pl

    def initialize(self, kwargs):
        self.__dict__.update(kwargs)
        req_param(self, ['backend', 'batch_size'])
        self.output = None
        self.berror = None

    def initialize_local(self):
        req_param(self, ['nifm', 'ifmshape', 'fshape'])

        opt_param(self, ['ofmlocs', 'links'])
        opt_param(self, ['berrorbuf', 'outputbuf'])

        opt_param(self, ['nofm'], self.nifm)
        opt_param(self, ['pooling'], False)
        opt_param(self, ['stride'], 1)
        opt_param(self, ['pad'], 0)

        self.fheight, self.fwidth = self.fshape
        self.ifmheight, self.ifmwidth = self.ifmshape
        self.ofmheight = int(mt.ceil(
            (self.ifmheight - self.fheight + 1 + 2. * self.pad) / self.stride))
        self.ofmwidth = int(mt.ceil(
            (self.ifmwidth - self.fwidth + 1 + 2. * self.pad) / self.stride))
        self.pad = -self.pad
        self.ofmshape = (self.ofmheight, self.ofmwidth)
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth
        self.fsize = self.fheight * self.fwidth * self.nifm
        self.fpsize = self.fheight * self.fwidth
        self.nout = self.nofm * self.ofmsize
        logger.debug('name=%s, ifmshape[0]=%d, ifmshape[1]=%d, nifm=%d, '
                     'ofmshape[0]=%d, ofmshape[1]=%d', self.name,
                     self.ifmshape[0], self.ifmshape[1], self.nifm,
                     self.ofmshape[0], self.ofmshape[1])
        if isinstance(self.backend, CPU):
            self.make_aux_buffers(self.nifm, self.ifmshape, self.nofm,
                                  self.ofmshape, self.fshape, self.stride)

    def __str__(self):
        if self.is_local:
            ionumstr = "({} x {}) x {} inputs, ({} x {}) x {} nodes".format(
                       self.ifmshape[0], self.ifmshape[1], self.nifm,
                       self.ofmshape[0], self.ofmshape[1], self.nofm)
        else:
            ionumstr = "{nin} inputs, {nout} nodes".format(
                       nin=self.nin, nout=self.nout)

        return ("Layer {lyr_tp} {lyr_nm}: {ionum}, {act_nm} act_fn, "
                "utilizing {be_nm} backend\n\t".format
                (lyr_tp=self.__class__.__name__,
                 lyr_nm=self.name, ionum=ionumstr,
                 act_nm=self.activation.__class__.__name__,
                 be_nm=self.backend.__class__.__name__))

    def allocate_output_bufs(self):
        make_zbuf = self.backend.zeros
        opt_param(self, ['out_shape'], (self.nout, self.batch_size))
        opt_param(self, ['berr_shape'], (self.nin, self.batch_size))

        self.output = make_zbuf(self.out_shape, self.output_dtype)

        if self.activation is not None:
            self.pre_act = make_zbuf(self.out_shape, self.pre_act_dtype)
        else:
            self.pre_act = self.output

        self.berror = None
        if (self.prev_layer is not None and not self.prev_layer.is_data):
            self.berror = make_zbuf(self.berr_shape, self.berror_dtype)

    def make_links(self, nifm, ifmsize, ifmshape, ofmshape, fshape, stride):
        # Figure out local connections to the previous layer.
        links = []
        for row in range(ofmshape[0]):
            for col in range(ofmshape[1]):
                # This variable tracks the top left corner of
                # the receptive field.
                src = (row * ifmshape[1] + col) * stride
                indlist = []
                for frow in range(fshape[0]):
                    start = src + frow * ifmshape[1]
                    indlist.extend(range(start, start + fshape[1]))
                fminds = np.array(indlist)
                if self.pooling is False:
                    for ifm in range(1, nifm):
                        indlist.extend(list(fminds + ifm * ifmsize))
                links.append(indlist)
        self.links = np.array(links, dtype='int32')

    def make_aux_buffers(self, nifm, ifmshape, nofm, ofmshape, fshape, stride):
        buf_size = self.batch_size * nifm
        if (self.prev_layer is not None and not self.prev_layer.is_data):
            self.berrorbuf = self.backend.empty((self.ifmsize, buf_size))

        ofmstarts = np.arange(0, (self.ofmsize * nofm), self.ofmsize)
        self.ofmlocs = np.empty((self.ofmsize, nofm), dtype='int32')
        for dst in range(self.ofmsize):
            self.ofmlocs[dst] = ofmstarts + dst
        self.make_links(nifm, self.ifmsize, ifmshape, ofmshape, fshape, stride)

        if self.pooling is True:
            self.outputbuf = self.backend.empty((self.ofmsize, buf_size))
            if self.op == 'max':
                self.tempbuf = np.empty(
                    (self.ofmsize, self.batch_size * nifm), dtype='int32')
            elif self.op == 'l2':
                self.tempbuf = self.backend.empty(
                    (self.fpsize, self.batch_size * nifm))

    def fprop(self, inputs):
        raise NotImplementedError('This class should not be instantiated.')

    def bprop(self, error):
        raise NotImplementedError('This class should not be instantiated.')

    def update(self, epoch):
        pass

    def set_train_mode(self, mode):
        pass


class CostLayer(Layer):

    def __init__(self, **kwargs):
        self.is_cost = True
        self.nout = 1
        super(CostLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        super(CostLayer, self).initialize(kwargs)
        req_param(self, ['cost', 'ref_layer'])
        opt_param(self, ['ref_label'], 'targets')
        self.targets = None
        self.cost.olayer = self.prev_layer
        self.cost.initialize(kwargs)
        self.berror = self.cost.get_berrbuf()

    def __str__(self):
        return ("Layer {lyr_nm}: {nin} nodes, {cost_nm} cost_fn, "
                "utilizing {be_nm} backend\n\t".format
                (lyr_nm=self.name, nin=self.nin,
                 cost_nm=self.cost.__class__.__name__,
                 be_nm=self.backend.__class__.__name__))

    def fprop(self, inputs):
        pass

    def bprop(self, error):
        # Since self.berror already pointing to destination of act gradient
        # we just have to scale by mini-batch size
        if self.ref_layer is not None:
            self.targets = getattr(self.ref_layer, self.ref_label)
        # if self.ref_label != 'targets':
        #     print self.targets.shape
        self.cost.apply_derivative(self.targets)
        self.backend.divide(self.berror, self.batch_size, out=self.berror)

    def get_cost(self):
        result = self.cost.apply_function(self.targets)
        return self.backend.divide(result, self.batch_size, result)


class DataLayer(Layer):

    def __init__(self, **kwargs):
        self.is_data = True
        super(DataLayer, self).__init__(**kwargs)
        req_param(self, ['dataset'])

    def initialize(self, kwargs):
        super(DataLayer, self).initialize(kwargs)
        self.batch_idx = 0
        if self.is_local is True:
            req_param(self, ['nofm', 'ofmshape'])
            self.nout = self.nofm * self.ofmshape[0] * self.ofmshape[1]
        else:
            req_param(self, ['nout'])

    def __str__(self):
        if self.is_local:
            ionumstr = "({} x {}) x {} nodes".format(
                       self.ofmshape[0], self.ofmshape[1], self.nofm)
        else:
            ionumstr = "{nout} nodes".format(nout=self.nout)

        return ("Layer {lyr_tp} {lyr_nm}: {ionum}\n\t".format
                (lyr_tp=self.__class__.__name__,
                 lyr_nm=self.name, ionum=ionumstr))

    def set_previous_layer(self, pl):
        pass

    def has_more_data(self):
        return True if (self.batch_idx < self.num_batches) else False

    def reset_counter(self):
        self.batch_idx = 0

    def fprop(self, inputs):
        ds = self.dataset
        self.output, self.targets = ds.get_mini_batch(self.batch_idx)
        self.batch_idx += 1

    def bprop(self, error):
        pass

    def has_set(self, setname):
        return self.dataset.has_set(setname)

    def use_set(self, setname, predict=False):
        ds = self.dataset
        self.num_batches = ds.init_mini_batch_producer(
            batch_size=self.batch_size,
            setname=setname,
            predict=predict)
        self.batch_idx = 0

    def cleanup(self):
        ds = self.dataset
        # delete helper queues if any
        ds.del_mini_batch_producer()


class ActivationLayer(Layer):

    """
    Just applies an activation to the inputs.
    """

    def set_previous_layer(self, pl):
        if pl.is_local:
            self.is_local = True
            self.ifmshape = pl.ofmshape
            self.nifm = pl.nofm
            self.nin = pl.nofm * pl.ofmshape[0] * pl.ofmshape[1]
        else:
            self.nin = pl.nout
        self.prev_layer = pl

    def initialize(self, kwargs):
        super(ActivationLayer, self).initialize(kwargs)
        req_param(self, ['activation'])
        self.nout = self.nin
        self.allocate_output_bufs()

    def fprop(self, inputs):
        self.pre_act[:] = inputs
        self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        if self.skip_act is False:
            self.backend.multiply(error, self.pre_act, out=error)
        if self.berror is not None:
            self.berror[:] = error


class WeightLayer(Layer):

    def initialize(self, kwargs):
        super(WeightLayer, self).initialize(kwargs)
        req_param(self, ['weight_init', 'lrule_init'])
        opt_param(self, ['accumulate'], False)

    def allocate_param_bufs(self):
        make_ebuf = self.backend.empty
        self.weights = self.backend.gen_weights(
            self.weight_shape, self.weight_init, self.weight_dtype)
        self.weight_updates = make_ebuf(self.weight_shape, self.updates_dtype)

        self.use_biases = 'bias_init' in self.weight_init
        opt_param(self, ['brule_init'], None)
        if self.use_biases is True:
            self.biases = make_ebuf(self.bias_shape, self.weight_dtype)
            self.biases.fill(self.weight_init['bias_init'])
            self.bias_updates = make_ebuf(self.bias_shape, self.updates_dtype)
            self.params = [self.weights, self.biases]
            self.updates = [self.weight_updates, self.bias_updates]
        else:
            self.params = [self.weights]
            self.updates = [self.weight_updates]

        if self.accumulate:
            self.utemp = map(lambda x: make_ebuf(x.shape, self.updates_dtype),
                             self.updates)

        self.learning_rule = self.init_learning_rule(self.lrule_init)
        self.bias_rule = None
        if self.brule_init is not None and self.use_biases:
            self.bias_rule = self.init_learning_rule(self.brule_init)
            self.bias_rule.allocate_state([self.bias_updates])
            self.learning_rule.allocate_state([self.weight_updates])
        else:
            self.learning_rule.allocate_state(self.updates)

    def update(self, epoch):
        if self.bias_rule is None:
            self.learning_rule.apply_rule(self.params, self.updates, epoch)
        else:
            self.learning_rule.apply_rule([self.weights],
                                          [self.weight_updates], epoch)
            self.bias_rule.apply_rule([self.biases],
                                      [self.bias_updates], epoch)

        if self.accumulate:
            for upm in self.updates:
                upm.fill(0.0)

    def normalize_weights(self, wts):
        norms = self.backend.norm(wts, order=2, axis=1)
        self.backend.divide(wts, norms.reshape((norms.shape[0], 1)), out=wts)

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


class FCLayer(WeightLayer):

    def initialize(self, kwargs):
        super(FCLayer, self).initialize(kwargs)
        req_param(self, ['nin', 'nout'])

        self.weight_shape = (self.nout, self.nin)
        self.bias_shape = (self.nout, 1)

        self.allocate_output_bufs()
        self.allocate_param_bufs()

    def fprop(self, inputs):
        self.backend.fprop_fc(out=self.pre_act, inputs=inputs,
                              weights=self.weights)
        if self.use_biases is True:
            self.backend.add(self.pre_act, self.biases, out=self.pre_act)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        inputs = self.prev_layer.output
        if self.activation is not None and self.skip_act is False:
            self.backend.multiply(error, self.pre_act, out=error)

        if self.berror is not None:
            self.backend.bprop_fc(out=self.berror, weights=self.weights,
                                  deltas=error)

        upm = self.utemp if self.accumulate else self.updates

        self.backend.update_fc(out=upm[0], inputs=inputs, deltas=error)
        if self.use_biases is True:
            self.backend.sum(error, axes=1, out=upm[1])

        if self.accumulate:
            self.backend.add(upm[0], self.updates[0], out=self.updates[0])
            if self.use_biases is True:
                self.backend.add(upm[1], self.updates[1], out=self.updates[1])


class PoolingLayer(Layer):

    """
    Generic pooling layer -- specify op = max, avg, or l2 in order to have
    it perform the desired pooling function
    """

    def __init__(self, **kwargs):
        self.is_local = True
        super(PoolingLayer, self).__init__(**kwargs)
        req_param(self, ['op'])

    def initialize(self, kwargs):
        super(PoolingLayer, self).initialize(kwargs)
        self.pooling = True
        self.tempbuf = None
        self.initialize_local()
        self.allocate_output_bufs()

    def fprop(self, inputs):
        self.backend.fprop_pool(out=self.output, inputs=inputs, op=self.op,
                                ofmshape=self.ofmshape, ofmlocs=self.tempbuf,
                                fshape=self.fshape, ifmshape=self.ifmshape,
                                links=self.links, nifm=self.nifm, padding=0,
                                stride=self.stride, fpropbuf=self.outputbuf)

    def bprop(self, error):
        inputs = self.prev_layer.output
        if self.berror is not None:
            self.backend.bprop_pool(out=self.berror, fouts=self.output,
                                    inputs=inputs, deltas=error, op=self.op,
                                    ofmshape=self.ofmshape,
                                    ofmlocs=self.tempbuf, fshape=self.fshape,
                                    ifmshape=self.ifmshape, links=self.links,
                                    nifm=self.nifm, padding=0,
                                    stride=self.stride,
                                    bpropbuf=self.berrorbuf)


class ConvLayer(WeightLayer):

    """
    Convolutional layer.
    """

    def __init__(self, **kwargs):
        self.is_local = True
        super(ConvLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        super(ConvLayer, self).initialize(kwargs)
        self.initialize_local()
        if self.pad != 0 and isinstance(self.backend, CPU):
            raise NotImplementedError('pad != 0, for CPU backend in ConvLayer')
        opt_param(self, ['local_conv'], False)

        if self.local_conv is False:
            self.weight_shape = (self.fsize, self.nofm)
        else:
            self.weight_shape = (self.fsize * self.ofmsize, self.nofm)
        self.bias_shape = (self.nout, 1)

        self.allocate_output_bufs()
        self.allocate_param_bufs()
        opt_param(self, ['prodbuf', 'bpropbuf', 'updatebuf'], None)
        if isinstance(self.backend, CPU):
            self.prodbuf = self.backend.empty((self.nofm, self.batch_size))
            self.bpropbuf = self.backend.empty((self.fsize, self.batch_size))
            self.updatebuf = self.backend.empty(self.weights.shape)

    def fprop(self, inputs):
        self.backend.fprop_conv(out=self.pre_act, inputs=inputs,
                                weights=self.weights, ofmshape=self.ofmshape,
                                ofmlocs=self.ofmlocs, ifmshape=self.ifmshape,
                                links=self.links, nifm=self.nifm,
                                padding=self.pad, stride=self.stride,
                                ngroups=1, fpropbuf=self.prodbuf,
                                local=self.local_conv)
        if self.use_biases is True:
            self.backend.add(self.pre_act, self.biases, out=self.pre_act)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        inputs = self.prev_layer.output
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)
        if self.berror is not None:
            self.backend.bprop_conv(out=self.berror, weights=self.weights,
                                    deltas=error, ofmshape=self.ofmshape,
                                    ofmlocs=self.ofmlocs,
                                    ifmshape=self.ifmshape, links=self.links,
                                    padding=self.pad, stride=self.stride,
                                    nifm=self.nifm, ngroups=1,
                                    bpropbuf=self.bpropbuf,
                                    local=self.local_conv)

        upm = self.utemp if self.accumulate else self.updates

        self.backend.update_conv(out=upm[0], inputs=inputs,
                                 weights=self.weights, deltas=error,
                                 ofmshape=self.ofmshape, ofmlocs=self.ofmlocs,
                                 ifmshape=self.ifmshape, links=self.links,
                                 nifm=self.nifm, padding=self.pad,
                                 stride=self.stride, ngroups=1,
                                 fwidth=self.fwidth, updatebuf=self.updatebuf,
                                 local=self.local_conv)

        if self.use_biases is True:
            self.backend.sum(error, axes=1, out=upm[1])
        if self.accumulate:
            self.backend.add(upm[0], self.updates[0], out=self.updates[0])
            if self.use_biases is True:
                self.backend.add(upm[1], self.updates[1], out=self.updates[1])


class DropOutLayer(Layer):

    """
    Dropout layer randomly kills activations from being passed on at each
    fprop call.
    Uses parameter 'keep' as the threshhold above which to retain activation.
    During training, the mask is applied, but during inference, we switch
    off the random dropping.
    Make sure to set train mode to False during inference.
    """

    def initialize(self, kwargs):
        opt_param(self, ['keep'], 0.5)
        super(DropOutLayer, self).initialize(kwargs)
        if self.prev_layer.is_local:
            self.is_local = True
            self.nifm = self.nofm = self.prev_layer.nofm
            self.ifmshape = self.ofmshape = self.prev_layer.ofmshape
        self.nout = self.nin
        self.keepmask = self.backend.empty((self.nin, self.batch_size))
        self.train_mode = True
        self.allocate_output_bufs()

    def fprop(self, inputs):
        if (self.train_mode):
            self.backend.fill_uniform_thresh(self.keepmask, self.keep)
            self.backend.multiply(self.keepmask, self.keep, out=self.keepmask)
            self.backend.multiply(inputs, self.keepmask, out=self.output)
        else:
            self.backend.multiply(inputs, self.keep, out=self.output)

    def bprop(self, error):
        if self.berror is not None:
            self.backend.multiply(error, self.keepmask, out=self.berror)

    def set_train_mode(self, mode):
        self.train_mode = mode


class CompositeLayer(Layer):

    """
    Abstract layer parent for Branch and List layer that deals with sublayer
    list
    """

    def initialize(self, kwargs):
        super(CompositeLayer, self).initialize(kwargs)
        req_param(self, ['sublayers'])
        for l in self.sublayers:
            l.initialize(kwargs)

    def update(self, epoch):
        for l in self.sublayers:
            l.update(epoch)

    def set_train_mode(self, mode):
        for sublayer in self.sublayers:
            sublayer.set_train_mode(mode)


class BranchLayer(CompositeLayer):

    """
    Branch layer is composed of a list of other layers concatenated with one
    another
    during fprop, it concatenates the component outputs and passes it on
    during bprop, it splits the backward errors into the components and
        accumulates into a common berror
    """

    def set_previous_layer(self, pl):
        super(BranchLayer, self).set_previous_layer(pl)
        for l in self.sublayers:
            l.set_previous_layer(pl)

    def initialize(self, kwargs):
        super(BranchLayer, self).initialize(kwargs)

        self.nout = reduce(lambda x, y: x + y.nout, self.sublayers, 0)
        self.startidx = [0] * len(self.sublayers)
        self.endidx = [0] * len(self.sublayers)
        self.endidx[0] = self.sublayers[0].nout
        for i in range(1, len(self.sublayers)):
            self.endidx[i] = self.endidx[i - 1] + self.sublayers[i].nout
            self.startidx[i] = self.endidx[i - 1]

        self.allocate_output_bufs()

    def fprop(self, inputs):
        for (s_l, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            s_l.fprop(inputs)
            self.output[si:ei] = s_l.output

    def bprop(self, error):
        for (s_l, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            s_l.bprop(error[si:ei])

        if self.berror is not None:
            self.berror.fill(0.0)
            for subl in self.sublayers:
                self.backend.add(self.berror, subl.berror, out=self.berror)


class ListLayer(Layer):

    """
    List layer is composed of a list of other layers stacked on top of one
    another
    during fprop, it simply fprops along the chain
    during bprop, it splits the backward errors into the components and
        accumulates into a common berror
    """

    def set_previous_layer(self, pl):
        super(ListLayer, self).set_previous_layer(pl)
        for l in self.sublayers:
            l.set_previous_layer(pl)
            pl = l

    def initialize(self, kwargs):
        super(ListLayer, self).initialize(kwargs)
        self.output = self.sublayers[-1].output
        self.berror = self.sublayers[0].berror
        self.nout = self.sublayers[-1].nout
        if self.sublayers[-1].is_local is True:
            self.nofm = self.sublayers[-1].nofm
            self.ofmshape = self.sublayers[-1].ofmshape

    def fprop(self, inputs):
        y = inputs
        for l in self.sublayers:
            l.fprop(y)
            y = l.output

    def bprop(self, error):
        error = None
        for l in reversed(self.sublayers):
            l.bprop(error)


class CrossMapResponseNormLayer(Layer):

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

    def __init__(self, **kwargs):
        self.is_local = True
        self.stride = 1
        super(CrossMapResponseNormLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        req_param(self, ['ksize', 'alpha', 'beta'])
        self.alpha = self.alpha * 1.0 / self.ksize
        super(CrossMapResponseNormLayer, self).initialize(kwargs)
        self.nout = self.nin
        self.ofmshape, self.nofm = self.ifmshape, self.nifm
        self.allocate_output_bufs()
        self.tempbuf = None
        if self.berror is not None and isinstance(self.backend, CPU):
            self.tempbuf = self.backend.empty(
                (self.ifmshape[0], self.ifmshape[1], self.batch_size))

    def fprop(self, inputs):
        self.backend.fprop_cmrnorm(out=self.output, inputs=inputs,
                                   ifmshape=self.ifmshape, nifm=self.nifm,
                                   ksize=self.ksize, alpha=self.alpha,
                                   beta=self.beta)

    def bprop(self, error):
        inputs = self.prev_layer.output
        if self.berror is not None:
            self.backend.bprop_cmrnorm(out=self.berror, fouts=self.output,
                                       inputs=inputs, deltas=error,
                                       ifmshape=self.ifmshape, nifm=self.nifm,
                                       ksize=self.ksize, alpha=self.alpha,
                                       beta=self.beta, bpropbuf=self.tempbuf)


class LocalContrastNormLayer(CrossMapResponseNormLayer):

    """
    Local contrast normalization.
    """

    def initialize(self, kwargs):
        super(LocalContrastNormLayer, self).initialize(kwargs)
        self.meandiffs = self.backend.empty(self.output.shape)
        self.denoms = self.backend.empty(self.output.shape)

        # Note dividing again is INTENTIONAL, since this is normalized by an
        # area not just a linear dimension
        self.alpha = self.alpha * 1.0 / self.ksize
        if self.stride != 1:
            raise NotImplementedError('stride != 1, in LocalContrastNormLayer')
        if self.ifmshape[0] != self.ifmshape[1]:
            raise NotImplementedError('non-square inputs not supported')

    def fprop(self, inputs):
        self.backend.fprop_lcnnorm(out=self.output, inputs=inputs,
                                   meandiffs=self.meandiffs,
                                   denoms=self.denoms, ifmshape=self.ifmshape,
                                   nifm=self.nifm, ksize=self.ksize,
                                   alpha=self.alpha, beta=self.beta)

    def bprop(self, error):
        if self.berror is not None:
            self.backend.bprop_lcnnorm(out=self.berror, fouts=self.output,
                                       deltas=error, meandiffs=self.meandiffs,
                                       denoms=self.denoms,
                                       ifmshape=self.ifmshape, nifm=self.nifm,
                                       ksize=self.ksize, alpha=self.alpha,
                                       beta=self.beta)


class CrossMapPoolingLayer(WeightLayer):

    """
    Pool input feature maps by computing a weighted sum of
    corresponding spatial locations across maps. This is
    equivalent to a 1x1 convolution.
    """

    def __init__(self, **kwargs):
        self.is_local = True
        super(CrossMapPoolingLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        self.fshape = (1, 1)
        super(CrossMapPoolingLayer, self).initialize(kwargs)
        req_param(self, ['nofm'])

        self.initialize_local()
        self.weight_shape = (self.nifm, self.nofm)
        self.allocate_output_bufs()
        self.allocate_param_bufs()
        opt_param(self, ['updatebuf'], None)
        if isinstance(self.backend, CPU):
            self.updatebuf = self.backend.empty((1, 1))

    def fprop(self, inputs):
        self.backend.fprop_cmpool(out=self.pre_act, inputs=inputs,
                                  weights=self.weights, ifmshape=self.ifmshape)
        if self.activation is not None:
            self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error):
        inputs = self.prev_layer.output
        if self.activation is not None:
            self.backend.multiply(error, self.pre_act, out=error)
        if self.berror is not None:
            self.backend.bprop_cmpool(out=self.berror, weights=self.weights,
                                      deltas=error, ifmshape=self.ifmshape)
        self.backend.update_cmpool(out=self.updates[0], inputs=inputs,
                                   deltas=error, ifmshape=self.ifmshape,
                                   updatebuf=self.updatebuf)
