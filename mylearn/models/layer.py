"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
from mylearn.transforms.gaussian import gaussian_filter
from mylearn.util.persist import YAMLable


logger = logging.getLogger(__name__)


class Layer(YAMLable):
    """
    Single NNet layer built to handle data from a particular backend

    Attributes:
        name (str): Used to identify this layer when logging.
        backend (mylearn.backends.backend.Backend): underlying type for stored
                                                    data parameters like
                                                    weights.
        batch_size (int): Number of examples presented at each iteration
        pos (int): The layers position (0-based)
        weights (mylearn.backends.backend.Tensor): weight values associated
                                                   with each node.
        activation (mylearn.transforms.activation.Activation): activation
                   function to apply to each node during a forward propogation
        nin (int): Number of inputs to this layer.
        nout (int): Number of outputs from this layer.
        output (mylearn.backends.backend.Tensor): final transformed output
                                                  values from this layer.
        pre_act (mylearn.backends.backend.Tensor): intermediate node values
                                                   from this layer prior
                                                   to applying activation
                                                   transform.
    """
    def __init__(self, name, backend, batch_size, pos, learning_rate, nin,
                 nout, activation, weight_init):
        self.name = name
        self.backend = backend
        self.activation = activation
        self.nin = nin
        self.nout = nout
        self.weights = self.backend.gen_weights((nout, nin), weight_init)
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
    Single NNet layer with no bias node
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
        self.activation.apply_both(self.backend, self.pre_act, self.output)

    def bprop(self, error, inputs, epoch, momentum):
        self.backend.multiply(error, self.pre_act, out=self.delta)
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


class LocalLayer(YAMLable):

    """
    Base class for locally connected layers.
    """

    def __init__(self, name, backend, batch_size, pos, learning_rate, nifm,
                 nofm, ifmshape, fshape, stride):
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
        self.nofm = nofm
        self.fsize = nifm * self.fheight * self.fwidth
        ofmstarts = backend.array(range(0, (self.ofmsize * nofm),
                                        self.ofmsize))
        ofmlocs = backend.zeros((self.ofmsize, nofm), dtype='i32')
        for dst in xrange(self.ofmsize):
            ofmlocs[dst, :] = ofmstarts + dst
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
            self.links[dst, :] = backend.array(colinds)
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
                 nifm, nofm, ifmshape, fshape, stride, weight_init, pretraining,
                 pretrain_learning_rate, sparsity, tied_weights):
        super(LocalFilteringLayer, self).__init__(name, backend, batch_size,
                                                  pos, learning_rate,
                                                  nifm, nofm, ifmshape, fshape,
                                                  stride)
        self.ifmsize = ifmshape[0] * ifmshape[1]
        self.nout = self.ofmsize * nofm
        self.output = backend.zeros((batch_size, self.nout))
        self.weights = self.backend.gen_weights((self.nout, self.fsize),
                                                weight_init)
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
        error = cost.apply_function(self.backend, self.defilter.output,
                                    inputs, self.defilter.temp)
        return error

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
        self.bpropbuf = prev.backend.zeros((prev.batch_size, 1))
        self.berror = prev.backend.zeros((prev.batch_size, prev.ofmsize))
        self.temp = [prev.backend.zeros(self.output.shape)]
        self.learning_rate = prev.pretrain_learning_rate
        self.backend = prev.backend
        self.rlinks = prev.rlinks
        self.prev = prev

    def fprop(self, inputs):
        self.backend.clear(self.output)
        for dst in xrange(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(inputs[:, dst:(dst + 1)],
                             self.weights[dst:(dst + 1)],
                             out=self.prodbuf)
            self.output[:, rflinks] += self.prodbuf

    def bprop(self, error, inputs, epoch, momentum):
        for dst in xrange(self.prev.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(error[:, rflinks],
                             self.weights[dst:(dst + 1)].T(),
                             out=self.bpropbuf)
            self.berror[:, dst:(dst+1)] = self.bpropbuf
            delta_slice = error[:, rflinks]
            self.backend.dot(self.output[:, dst:(dst+1)].T(), delta_slice,
                             out=self.updates[dst:(dst+1)])
        self.backend.multiply(self.updates,
                              self.backend.wrap(self.learning_rate),
                              out=self.updates)
        self.backend.subtract(self.weights, self.updates, out=self.weights)
        self.prev.normalize_weights(self.weights)


class PoolingLayer(YAMLable):

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

    def __init__(self, name, backend, batch_size, pos, nfm, ifmshape, fshape,
                 stride):
        super(LCNLayer, self).__init__(name, backend, batch_size, pos, 0.0,
                                       nfm, nfm, ifmshape, fshape, stride)
        self.nfm = nfm
        self.nin = nfm * self.ifmsize
        self.nout = self.nin
        self.output = backend.zeros((batch_size, self.nout))
        self.routput = self.output.reshape((batch_size,
                                            self.nfm,
                                            self.ifmheight,
                                            self.ifmwidth))
        self.filters = self.normalized_gaussian_filters(nfm, fshape)
        self.meanfm = self.backend.zeros((batch_size, self.ofmsize))
        self.rmeanfm = self.meanfm.reshape((batch_size,
                                            1,
                                            self.ofmheight,
                                            self.ofmwidth))
        assert (self.ifmheight - self.ofmheight) % 2 == 0
        assert (self.ifmwidth - self.ofmwidth) % 2 == 0

        self.prodbuf = backend.zeros((batch_size, 1))
        self.intermed = backend.zeros(self.output.shape)
        self.rintermed = self.intermed.reshape((batch_size,
                                                self.nfm,
                                                self.ifmheight,
                                                self.ifmwidth))
        # Compute co-ordinates for the mean feature map obtained by
        # convolving.
        self.start_row = (self.ifmheight - self.ofmheight) / 2
        self.end_row = self.start_row + self.ofmheight
        self.start_col = (self.ifmwidth - self.ofmwidth) / 2
        self.end_col = self.start_col + self.ofmwidth

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
        filters = self.backend.zeros((count, shape[0], shape[1]))
        filters[:] = single

        filters = filters.reshape((count * shape[0] * shape[1], 1))
        return filters

    def expand_image(self, exfm):
        # Fill the borders with duplicated rows/columns.
        exfm[:, :, :self.start_row, :] = (
            exfm[:, :, self.start_row:(self.start_row + 1), :])
        exfm[:, :, self.end_row:self.ifmheight, :] = (
            exfm[:, :, (self.end_row - 1):self.end_row, :])

        exfm[:, :, :, :self.start_col] = (
            exfm[:, :, :, self.start_col:(self.start_col + 1)])
        exfm[:, :, :, self.end_col:self.ifmwidth] = (
            exfm[:, :, :, (self.end_col - 1):self.end_col])

    def sub_normalize(self, inputs):
        # Convolve with gaussian filters to obtain a "mean" feature map.
        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(inputs.take(rflinks, axis=1), self.filters,
                             out=self.prodbuf)
            self.meanfm[:, dst:(dst + 1)] = self.prodbuf

        rinputs = inputs.reshape((self.batch_size, self.nfm,
                                  self.ifmheight, self.ifmwidth))
        self.backend.subtract(rinputs[:, :,
                              self.start_row:self.end_row,
                              self.start_col:self.end_col],
                              self.rmeanfm,
                              out=self.rintermed[:, :,
                              self.start_row:self.end_row,
                              self.start_col:self.end_col])
        self.expand_image(self.rintermed)

    def div_normalize(self):
        self.backend.multiply(self.intermed, self.intermed, out=self.output)
        self.backend.clear(self.meanfm)
        for dst in xrange(self.ofmsize):
            rflinks = self.rlinks[dst]
            self.backend.dot(self.output.take(rflinks, axis=1), self.filters,
                             out=self.prodbuf)
            self.meanfm[:, dst:(dst + 1)] = self.prodbuf
        self.backend.sqrt(self.meanfm, out=self.meanfm)
        mean = self.meanfm.mean()
        self.meanfm[self.meanfm < mean] = mean
        self.backend.divide(self.rintermed[:, :,
                            self.start_row:self.end_row,
                            self.start_col:self.end_col],
                            self.rmeanfm,
                            out=self.routput[:, :,
                            self.start_row:self.end_row,
                            self.start_col:self.end_col])
        self.expand_image(self.routput)

    def fprop(self, inputs):
        self.sub_normalize(inputs)
        self.div_normalize()

    def bprop(self, error, inputs, epoch, momentum):
        if self.pos > 0:
            self.berror[:] = error
