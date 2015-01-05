# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
import numpy as np
from neon.models.learning_rule import *
from neon.backends.cpu import CPU
from neon.transforms.gaussian import gaussian_filter
from neon.util.compat import MPI_INSTALLED, range
from neon.util.persist import YAMLable

logger = logging.getLogger(__name__)

def req_param(obj, paramlist):
    for param in paramlist:
        if not hasattr(obj, param):
            raise ValueError("required parameter %s missing for %s" % (param,
                              obj.name))

def opt_param(obj, paramlist, default_value=None):
    for param in paramlist:
        if not hasattr(obj, param):
            setattr(obj, param, default_value)


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

        opt_param(self, ['ofmlocs', 'links', 'rlinks'])
        opt_param(self, ['berrorbuf', 'outputbuf'])

        opt_param(self, ['nofm'], self.nifm)
        opt_param(self, ['pooling'], False)
        opt_param(self, ['stride'], 1)
        opt_param(self, ['pad'], 0)

        stride = self.stride
        batch_size = self.batch_size

        self.pad = -self.pad
        self.fheight, self.fwidth = self.fshape
        self.ifmheight, self.ifmwidth = self.ifmshape
        self.ofmheight = np.int(
            np.ceil(
                (self.ifmheight - self.fheight + 2. * self.pad) / stride)) + 1
        self.ofmwidth = np.int(
            np.ceil(
                (self.ifmwidth - self.fwidth + 2. * self.pad) / stride)) + 1
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

    def make_aux_buffers(self, nifm, ifmshape, nofm, ofmshape, fshape, stride):

        make_ebuf = self.backend.empty
        ofmsize = ofmshape[0] * ofmshape[1]
        ifmsize = ifmshape[0] * ifmshape[1]
        fsize = fshape[0] * fshape[1] * nifm
        fpsize = fshape[0] * fshape[1]
        buf_size = self.batch_size * nifm

        if (self.prev_layer is not None and not self.prev_layer.is_data):
            self.berrorbuf = make_ebuf((ifmsize, buf_size))

        ofmstarts = self.backend.array(range(0, (ofmsize * nofm),
                                        ofmsize)).raw()
        self.ofmlocs = make_ebuf((ofmsize, nofm), dtype='i32')
        for dst in range(ofmsize):
            self.ofmlocs[dst] = self.backend.wrap(ofmstarts + dst)
        if self.pooling is True:
            self.links = make_ebuf((ofmsize, fpsize), dtype='i32')
            self.outputbuf = make_ebuf((ofmsize, buf_size))
        else:
            self.links = make_ebuf((ofmsize, fsize), dtype='i32')
        # This variable tracks the top left corner of the receptive field.
        src = 0
        for dst in range(ofmsize):
            # Collect the column indices for the
            # entire receptive field.
            colinds = []
            for row in range(fshape[0]):
                start = src + row * ifmshape[1]
                colinds += range(start, start + fshape[1])
            fminds = colinds[:]
            if self.pooling is False:
                for ifm in range(1, nifm):
                    colinds += [x + ifm * ifmsize for x in fminds]

            if (src % ifmshape[1] + fshape[1] + stride) <= ifmshape[1]:
                # Slide the filter to the right by the stride value.
                src += stride
            else:
                # We hit the right edge of the input image.
                # Shift the filter down by one stride.
                src += stride * ifmshape[1] - src % ifmshape[1]
                assert src % ifmshape[1] == 0
            self.links[dst] = self.backend.array(colinds, dtype='i32')
        self.rlinks = self.links.raw()

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
        self.backend.divide(self.berror, self.backend.wrap(self.batch_size),
                            out=self.berror)

    def get_cost(self):
        return self.cost.apply_function(self.targets) / self.batch_size


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
        if ds.macro_batched:
            self.output, self.targets = ds.get_mini_batch(
                                            self.batch_size, 'training')
        else:
            self.output = ds.get_batch(self.inputs, self.batch_idx)
            self.targets = ds.get_batch(self.tgts, self.batch_idx)
        self.batch_idx += 1

    def bprop(self, error):
        pass

    def has_set(self, setname):
        if self.dataset.macro_batched:
            return True if (setname in ['train', 'validation']) else False
        else:
            inputs_dic = self.dataset.get_inputs(train=True, validation=True,
                                        test=True)
            return True if (setname in inputs_dic) else False

    def use_set(self, setname):
        ds = self.dataset
        if ds.macro_batched:
            sn = 'val' if (setname == 'validation') else setname
            endb = getattr(ds, 'end_' + sn + '_batch')
            startb = getattr(ds, 'start_' + sn + '_batch')
            nrecs = ds.output_batch_size * (endb - startb + 1)
            if startb == -1:
                nrecs = ds.max_file_index
            setattr(ds, 'cur_' + sn + '_macro_batch', startb)
            self.num_batches = int(np.ceil((nrecs + 0.0) / self.batch_size))
        else:
            self.inputs = ds.get_inputs(train=True, validation=True,
                                        test=True)[setname]
            self.tgts = ds.get_targets(train=True, validation=True,
                                        test=True)[setname]
            self.num_batches = len(self.inputs)
        self.batch_idx = 0


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
        opt_param(self,['accumulate'], False)

    def allocate_param_bufs(self):
        make_ebuf = self.backend.empty
        self.weights = self.backend.gen_weights(
                    self.weight_shape, self.weight_init, self.weight_dtype)
        self.weight_updates = make_ebuf(self.weight_shape, self.updates_dtype)

        self.use_biases = 'bias_init' in self.weight_init
        if self.use_biases:
            self.biases = make_ebuf(self.bias_shape, self.weight_dtype)
            self.backend.fill(self.biases, self.weight_init['bias_init'])
            self.bias_updates = make_ebuf(self.bias_shape, self.updates_dtype)
            self.params = [self.weights, self.biases]
            self.updates = [self.weight_updates, self.bias_updates]
        else:
            self.params = [self.weights]
            self.updates = [self.weight_updates]

        if self.accumulate:
            self.utemp = map(lambda x: make_ebuf(x.shape, self.updates_dtype),
                             self.updates)
        self.gen_learning_rule()

    def update(self, epoch):
        self.learning_rule.apply_rule(self.params, self.updates, epoch)
        if self.accumulate:
            for upm in self.updates:
                self.backend.clear(upm)

    def normalize_weights(self, wts):
        norms = self.backend.norm(wts, order=2, axis=1)
        self.backend.divide(wts, norms.reshape((norms.shape[0], 1)), out=wts)

    def gen_learning_rule(self):
        lrname = self.name + '_lr'
        if self.lrule_init['type'] == 'gradient_descent':
            self.learning_rule = GradientDescent(name=lrname,
                                    lr_params=self.lrule_init['lr_params'])
        elif self.lrule_init['type'] == 'gradient_descent_pretrain':
            self.learning_rule = GradientDescentPretrain(name=lrname,
                                    lr_params=self.lrule_init['lr_params'])
        elif self.lrule_init['type'] == 'gradient_descent_momentum':
            self.learning_rule = GradientDescentMomentum(name=lrname,
                                    lr_params=self.lrule_init['lr_params'])
        elif self.lrule_init['type'] == 'adadelta':
            self.learning_rule = AdaDelta(name=lrname,
                                    lr_params=self.lrule_init['lr_params'])
        else:
            raise AttributeError("invalid learning rule params specified")
        self.learning_rule.allocate_state(self.updates)


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
            self.backend.sum(error, axis=1, out=upm[1])

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
        self.initialize_local()
        self.tempbuf = None
        if self.op == 'max':
            self.tempbuf = self.backend.empty(
                (self.ofmsize, self.batch_size * self.nifm), dtype='i16')
        elif self.op == 'l2':
            self.tempbuf = self.backend.empty((self.fshape[0] * self.fshape[1],
                                           self.batch_size * self.nifm))
        self.allocate_output_bufs()
        assert self.fshape[0] * self.fshape[1] <= 2 ** 15

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
            # if isinstance(self.backend, CPU):
            #     self.weight_shape = (self.fsize, self.nofm * self.ofmsize)
            # else:
            #     self.weight_shape = (self.fsize * self.ofmsize, self.nofm)
        self.bias_shape = (self.nofm, 1)

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
                                links=self.rlinks, nifm=self.nifm,
                                padding=self.pad, stride=self.stride,
                                ngroups=1, fpropbuf=self.prodbuf,
                                local=self.local_conv)
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
                                    ifmshape=self.ifmshape, links=self.rlinks,
                                    padding=self.pad, stride=self.stride,
                                    nifm=self.nifm, ngroups=1,
                                    bpropbuf=self.bpropbuf,
                                    local=self.local_conv)

        upm = self.utemp if self.accumulate else self.updates
        self.backend.update_conv(out=upm[0], inputs=inputs,
                                 weights=self.weights, deltas=error,
                                 ofmshape=self.ofmshape, ofmlocs=self.ofmlocs,
                                 ifmshape=self.ifmshape, links=self.rlinks,
                                 nifm=self.nifm, padding=self.pad,
                                 stride=self.stride, ngroups=1,
                                 fwidth=self.fwidth, updatebuf=self.updatebuf,
                                 local=self.local_conv)

        if self.accumulate:
            self.backend.add(upm[0], self.updates[0], out=self.updates[0])


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
            self.backend.multiply(self.keepmask, self.backend.wrap(self.keep),
                                  out=self.keepmask)
            self.backend.multiply(inputs, self.keepmask, out=self.output)
        else:
            self.backend.multiply(inputs, self.backend.wrap(self.keep),
                                  out=self.output)

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
        self.startidx = [0]*len(self.sublayers)
        self.endidx = [0]*len(self.sublayers)
        self.endidx[0] = self.sublayers[0].nout
        for i in range(1, len(self.sublayers)):
            self.endidx[i] = self.endidx[i-1] + self.sublayers[i].nout
            self.startidx[i] = self.endidx[i-1]

        self.allocate_output_bufs()

    def fprop(self, inputs):
        for (s_l, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            s_l.fprop(inputs)
            self.output[si:ei] = s_l.output

    def bprop(self, error):
        inputs = self.prev_layer.output
        for (s_l, si, ei) in zip(self.sublayers, self.startidx, self.endidx):
            s_l.bprop(error[si:ei])

        if self.berror is not None:
            self.berror[:] = self.backend.wrap(0.0)
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
    def initialize(self, kwargs):
        self.__dict__.update(kwargs)
        req_param(self, ['backend', 'batch_size'])
        self.output = None
        self.berror = None

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

class LCNLayer(Layer):

    """
    Local contrast normalization.
    """
    def __init__(self, **kwargs):
        self.is_local = True
        super(LCNLayer, self).__init__(**kwargs)

    def initialize(self, kwargs):
        super(LCNLayer, self).initialize(kwargs)
        self.initialize_local()
        self.nout = self.nin
        self.ofmshape, self.nofm = self.ifmshape, self.nifm

        bsz = self.batch_size
        nifm = self.nifm

        self.filters = self.normalized_gaussian_filters(nifm, self.fshape)
        # Now filters is 1 x (nifm * fheight * fwidth)

        exifmheight = (self.ifmheight - 1) * self.stride + self.fheight
        exifmwidth = (self.ifmwidth - 1) * self.stride + self.fwidth
        self.exifmshape = (exifmheight, exifmwidth)
        self.exifmsize = exifmheight * exifmwidth

        # redo the link buffers since we'll be doing light-weight convolution
        # without redefining a new convolve layer
        self.make_aux_buffers(nifm=nifm, ifmshape=self.exifmshape, nofm=1,
                              ofmshape=self.ifmshape, fshape=self.fshape,
                              stride=self.stride)
        if isinstance(self.backend, CPU):
            self.prodbuf = self.backend.empty((1, bsz))

        assert (exifmheight - self.ifmheight) % 2 == 0
        assert (exifmwidth - self.ifmwidth) % 2 == 0
        self.start_row = (exifmheight - self.ifmheight) / 2
        self.start_col = (exifmwidth - self.ifmwidth) / 2

        ex_shape = (nifm * exifmheight * exifmwidth, bsz)
        rex_shape = (nifm, exifmheight, exifmwidth, bsz)
        self.exinputs = self.backend.empty(ex_shape)
        self.rexinputs = self.exinputs.reshape(rex_shape)

        cv_shape = (1 * self.ifmheight * self.ifmwidth * bsz)
        rcv_shape = (1, self.ifmheight, self.ifmwidth, bsz)
        self.meanfm = self.backend.empty(cv_shape)
        self.rmeanfm = self.meanfm.reshape(rcv_shape)

        out_shape = (nifm * self.ifmheight * self.ifmwidth * bsz)
        rout_shape = (nifm, self.ifmheight, self.ifmwidth, bsz)
        self.output = self.backend.empty(out_shape)
        self.routput = self.output.reshape(rout_shape)

        self.subout = self.backend.empty(out_shape)
        self.rsubout = self.subout.reshape(rout_shape)

        self.subtemp = self.backend.empty(out_shape)
        self.rsubtemp = self.subtemp.reshape(rout_shape)

        self.tempbuf = self.backend.empty((self.nofm, bsz))
        # Check position relative to other layer for this condition
        if (self.prev_layer is not None and not self.prev_layer.is_data):
            self.diverror = self.backend.empty((self.nin, bsz))

            self.exerror = self.backend.empty(ex_shape)
            self.rexerror = self.exerror.reshape(rex_shape)

            self.prodbuf = self.backend.empty((self.fsize, bsz))
            self.bprop_filters = self.backend.empty((nifm, 1, self.fsize))
            self.sqtemp = self.backend.empty(self.output.shape)
            for fm in range(nifm):
                self.bprop_filters[fm] = self.filters.copy()
                rfilter = self.bprop_filters[fm].reshape(
                    (nifm, self.fheight, self.fwidth))
                fm_filt = rfilter[fm, self.fheight / 2, self.fwidth / 2]
                self.backend.subtract(fm_filt, 1.0, fm_filt)

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

    def copy_to_inset(self, canvas, inset):
        r0 = self.start_row
        c0 = self.start_col
        canvas[:, r0:(canvas.shape[1] - r0), c0:(canvas.shape[2] - c0)] = inset

    def copy_from_inset(self, canvas):
        r0 = self.start_row
        c0 = self.start_col        
        return canvas[:, r0:(canvas.shape[1] - r0), c0:(canvas.shape[2] - c0)]

    def conv(self, inputs):
        # print self.filters.shape, inputs.shape, self.rlinks.shape, self.tempbuf.shape
        # print inputs.take(self.rlinks[0], axis=0).shape
        for dst in range(self.ifmsize):
            self.backend.dot(self.filters,
                             inputs.take(self.rlinks[dst], axis=0),
                             out=self.tempbuf)
            self.meanfm[self.ofmlocs[dst]] = self.tempbuf

    def fprop_sub_normalize(self, inputs):
        rinputs = inputs.reshape(self.routput.shape)
        self.copy_to_inset(self.rexinputs, rinputs)

        # Convolve with gaussian filters to obtain a "mean" feature map.
        self.conv(self.exinputs)
        self.backend.subtract(rinputs, self.rmeanfm, out=self.rsubout)

    def fprop_div_normalize(self):
        self.backend.multiply(self.subout, self.subout, out=self.subtemp)
        self.copy_to_inset(self.rexinputs, self.rsubtemp)
        self.conv(self.exinputs)

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
        self.berror = self.copy_from_inset(self.rexerror)
        self.berror = self.berror.reshape((self.nin, self.batch_size))

    def bprop_sub_normalize(self, error, inputs):
        self.backend.clear(self.exerror)
        for fm in range(self.nifm):
            for dst in range(self.ofmsize):
                rflinks = self.rlinks[dst]
                loc = self.ofmlocs[dst].raw() + self.ofmsize * fm
                filt = self.bprop_filters[fm]
                self.backend.multiply(error[loc], filt.transpose(),
                                      out=self.prodbuf)
                exerror_slice = self.exerror[rflinks]
                self.backend.subtract(exerror_slice, self.prodbuf,
                                      exerror_slice)
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
            for dst in range(self.ofmsize):
                # self.ofmlocs is over 1 fm only
                loc = self.ofmlocs[dst].raw() + self.ofmsize * fm
                divout = self.output.take(loc, axis=0)
                subout = self.subout.take(loc, axis=0)
                assert divout[subout.raw() == 0].sum() == 0
                subout[subout.raw() == 0.0] = 1.0
                self.backend.divide(divout, subout, out=divout)

                rflinks = self.rlinks[dst]
                self.copy_to_inset(self.rexinputs, self.rsubtemp)
                rrexinputs = self.rexinputs.reshape(
                    (self.nifm * self.exifmsize, self.batch_size))
                frame = rrexinputs.take(rflinks, axis=0)
                self.backend.multiply(frame, self.filters.transpose(),
                                      out=frame)
                self.backend.multiply(frame, self.diverror[loc], out=frame)
                rframe = frame.reshape((self.nifm, self.fheight, self.fwidth,
                                        self.batch_size))
                # this is working on the g2/y2 term
                rframe_slice = rframe[fm:(fm + 1), self.fheight / 2,
                                      self.fwidth / 2]
                self.backend.subtract(rframe_slice, divout, rframe_slice)
                self.backend.multiply(error[loc], frame, out=frame)
                exerror_slice = self.exerror[rflinks]
                self.backend.subtract(exerror_slice, frame, exerror_slice)
        self.reshape_error()

    def bprop(self, error, inputs):
        if (self.prev_layer is not None and not self.prev_layer.is_data):
            # note: have to account for halos + padding after each step
            self.bprop_div_normalize(error, inputs)
            self.bprop_sub_normalize(self.berror, inputs)

    def bprop_fast(self, error, inputs):
        """
        An incorrect, but much faster version of backprop.
        """
        if (self.prev_layer is not None and not self.prev_layer.is_data):
            self.berror[:] = error


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
        self.fshape = (1,1)
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
