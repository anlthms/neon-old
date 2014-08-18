"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging
from mylearn.transforms.gaussian import gaussian_filter


logger = logging.getLogger(__name__)


class Layer(object):

    """
    Single NNet layer built to handle data from a particular backend
    """

    def __init__(self, name, backend, nin, nout, activation, weight_init):
        self.name = name
        self.backend = backend
        self.weights = self.backend.gen_weights((nout, nin), weight_init)
        self.activation = activation
        self.nin = nin
        self.nout = nout
        self.velocity = self.backend.zeros(self.weights.shape)
        self.output = None
        self.pre_act = None

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
        self.pre_act = self.backend.dot(inputs, self.weights.T())
        self.output = self.activation.apply_function(self.pre_act)

    def bprop(self, error):
        self.delta = error * self.activation.apply_derivative(self.pre_act)

    def update(self, inputs, epsilon, epoch, momentum):
        inputs = self.backend.append_bias(inputs)
        momentum_coef = self.backend.get_momentum_coef(epoch, momentum)
        self.velocity = (momentum_coef * self.velocity -
                         epsilon * self.backend.dot(self.delta.T(), inputs))
        self.weights += self.velocity

    def error(self):
        return self.backend.dot(self.delta, self.weights.take(
                                range(self.weights.shape[1] - 1), axis=1))


class LayerWithNoBias(Layer):

    """
    Single NNet layer with no bias node - temporary code for testing purposes.
    """

    def fprop(self, inputs):
        self.pre_act = self.backend.dot(inputs, self.weights.T())
        self.output = self.activation.apply_function(self.pre_act)

    def update(self, inputs, epsilon, epoch, momentum):
        self.weights -= epsilon * self.backend.dot(self.delta.T(), inputs)

    def error(self):
        return self.backend.dot(self.delta, self.weights)


class RBMLayer(Layer):

    """
    CD1 training layer for RBM
    """

    def positive(self, inputs):
        """
        Positive / upward pass of the CD1 RBM

        Arguments:
           inputs (mylearn.datasets.dataset.Dataset): dataset upon which to operate
        """
        inputs = self.backend.append_bias(inputs)
        self.pre_act = self.backend.dot(inputs, self.weights.T())
        self.p_hid_plus = self.activation.apply_function(self.pre_act)
        self.p_plus = self.backend.dot(self.p_hid_plus.T(), inputs)
        self.random_numbers = self.backend.uniform(size=self.p_hid_plus.shape)
        self.s_hid_plus = self.p_hid_plus > self.random_numbers

    def negative(self, inputs):
        """
        Negative / downward pass of the CD1 RBM

        Arguments:
           inputs (mylearn.datasets.dataset.Dataset): dataset upon which to operate
        """
        self.pre_act = self.backend.dot(self.s_hid_plus, self.weights)
        self.x_minus = self.activation.apply_function(self.pre_act)
        self.pre_act = self.backend.dot(self.x_minus, self.weights.T())
        self.p_hid_minus = self.activation.apply_function(self.pre_act)
        self.p_minus = self.backend.dot(self.p_hid_minus.T(), self.x_minus)

    def update(self, epsilon, epoch, momentum):
        """ 
        CD1 weight update 

        Arguments:
            epsilon: step size
            epoch: not used, for future compatibility
            momentum: not used, for future compatibility
        """
        self.weights += epsilon * (self.p_plus - self.p_minus)
        # epoch, momentum?

    def error(self, inputs):
        pass
        # return ((inputs-self.x_minus)**2).mean()


class AELayer(object):

    """
    Single NNet layer built to handle data from a particular backend used
    in an Autoencoder.
    TODO: merge with generic Layer above.
    """

    def __init__(self, name, backend, nin, nout, activation, weight_init,
                 weights=None):
        self.name = name
        self.backend = backend
        if weights is None:
            self.weights = self.backend.gen_weights((nout, nin), weight_init)
        else:
            self.weights = weights
        self.activation = activation
        self.nin = nin
        self.nout = nout
        self.output = None
        self.pre_act = None

    def __str__(self):
        return ("Layer %s: %d inputs, %d nodes, %s act_fn, "
                "utilizing %s backend\n\t"
                "y: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "z: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
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
                 self.backend.max(self.weights)))

    def fprop(self, inputs):
        self.pre_act = self.backend.dot(inputs, self.weights.T())
        self.output = self.activation.apply_function(self.pre_act)

    def bprop(self, error):
        self.delta = error * self.activation.apply_derivative(self.pre_act)

    def update(self, inputs, epsilon, epoch):
        self.weights -= epsilon * self.backend.dot(self.delta.T(), inputs)

    def error(self):
        return self.backend.dot(self.delta, self.weights)


class LocalLayer(object):

    """
    Base class for locally connected layers.
    """

    def __init__(self, name, backend, batch_size, nifm, ifmshape, fshape,
                 stride):
        self.name = name
        self.backend = backend
        self.ifmheight, self.ifmwidth = ifmshape
        self.fheight, self.fwidth = fshape
        self.batch_size = batch_size

        self.ofmheight = (self.ifmheight - self.fheight) / stride + 1
        self.ofmwidth = (self.ifmwidth - self.fwidth) / stride + 1
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = self.ofmheight * self.ofmwidth

        self.nifm = nifm
        self.fsize = nifm * self.fheight * self.fwidth
        # Figure out the connections with the previous layer.
        self.links = backend.zeros((self.ofmsize, self.fsize), dtype='i32')
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
            self.links[dst, :] = backend.array(colinds)
        self.rlinks = self.links.raw()

    def bprop(self, error):
        self.delta = error


class ConvLayer(LocalLayer):

    """
    Convolutional layer.
    """

    def __init__(self, name, backend, batch_size, nifm,
                 ifmshape, fshape, nfilt, stride, weight_init):
        super(ConvLayer, self).__init__(name, backend, batch_size, nifm,
                                        ifmshape, fshape, stride)
        self.nout = self.ofmsize * nfilt
        self.nfilt = nfilt
        self.weights = self.backend.gen_weights((nfilt, self.fsize),
                                                weight_init)
        self.output = backend.zeros((batch_size, self.nout))
        ofmstarts = self.backend.array(range(0, (self.ofmsize * nfilt),
                                             self.ofmsize))
        ofmlocs = backend.zeros((self.ofmsize, nfilt), dtype='i32')
        for dst in range(self.ofmsize):
            ofmlocs[dst, :] = ofmstarts + dst
        self.rofmlocs = ofmlocs.raw()

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
        for dst in range(self.ofmsize):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            rflinks = self.rlinks[dst]
            prod = self.backend.dot(inputs.take(rflinks, axis=1),
                                    self.weights.T())
            self.output[:, self.rofmlocs[dst]] = prod

    def update(self, inputs, epsilon, epoch, momentum):
        wsums = self.backend.zeros(self.weights.shape)
        for dst in range(self.ofmsize):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = self.rlinks[dst]
            delta_slice = self.delta.take(self.rofmlocs[dst], axis=1)
            prod = self.backend.dot(delta_slice.T(), inputs.take(rflinks,
                                                                 axis=1))
            wsums.add(prod)
        # Update the filters after summing the weight updates.
        self.weights.sub(epsilon * wsums)

    def error(self):
        berror = self.backend.zeros((self.batch_size,
                                     self.ifmheight * self.ifmwidth *
                                     self.nifm))
        for dst in range(self.ofmsize):
            res = self.backend.dot(self.delta.take(self.rofmlocs[dst], axis=1),
                                   self.weights)
            rflinks = self.rlinks[dst]
            res.add(berror.take(rflinks, axis=1))
            berror[:, rflinks] = res
        return berror


class LocalFilteringLayer(LocalLayer):

    """
    Local filtering layer. This is very similar to ConvLayer, but the weights
    are not shared.
    """

    def __init__(self, name, backend, batch_size,
                 nifm, ifmshape, fshape, stride, weight_init):
        super(LocalFilteringLayer, self).__init__(name, backend, batch_size,
                                                  nifm, ifmshape, fshape,
                                                  stride)
        self.nout = self.ofmsize
        self.output = backend.zeros((batch_size, self.nout))
        self.weights = self.backend.gen_weights((self.ofmsize, self.fsize),
                                                weight_init)

    def __str__(self):
        return ("LocalFilteringLayer %s: %d ifms, "
                "utilizing %s backend\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nifm,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights)))

    def fprop(self, inputs):
        for dst in range(self.ofmsize):
            rflinks = self.rlinks[dst]
            # We use a different filter for each receptive field.
            prod = self.backend.dot(inputs.take(rflinks, axis=1),
                                    self.weights[dst].T())
            self.output[:, dst] = prod

    def update(self, inputs, epsilon, epoch, momentum):
        updates = self.backend.zeros(self.weights.shape)
        for dst in range(self.ofmsize):
            rflinks = self.rlinks[dst]
            delta_slice = self.delta.take(dst, axis=1)
            updates[dst] = self.backend.dot(delta_slice.T(),
                                            inputs.take(rflinks, axis=1))
        self.weights.sub(epsilon * updates)

    def error(self):
        berror = self.backend.zeros((self.batch_size,
                                     self.ifmheight * self.ifmwidth *
                                     self.nifm))
        for dst in range(self.ofmsize):
            # Use the same filter that was used for forward propagation
            # of this receptive field.
            res = self.backend.dot(self.delta.take(range(dst, dst + 1),
                                                   axis=1),
                                   self.weights[dst:(dst + 1)])
            rflinks = self.rlinks[dst]
            res.add(berror.take(rflinks, axis=1))
            berror[:, rflinks] = res
        return berror


class PoolingLayer(object):

    """
    Base class for pooling layers.
    """

    def __init__(self, name, backend, batch_size, nfm, ifmshape, pshape,
                 stride):
        self.name = name
        self.backend = backend
        self.nfm = nfm
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.pheight, self.pwidth = pshape
        self.psize = self.pheight * self.pwidth
        self.batch_size = batch_size

        ofmheight = (self.ifmheight - self.pheight) / stride + 1
        ofmwidth = (self.ifmwidth - self.pwidth) / stride + 1
        self.ofmsize = ofmheight * ofmwidth
        self.nin = nfm * self.ifmsize

        # Figure out the possible connections with the previous layer.
        # Each unit in this layer could be connected to any one of
        # self.psize units in the previous layer.
        self.links = backend.zeros((self.ofmsize, self.psize), dtype='i32')
        # This variable tracks the top left corner of the receptive field.
        src = 0
        for dst in range(self.ofmsize):
            colinds = []
            # Collect the column indices for the
            # entire receptive field.
            for row in range(self.pheight):
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

    def bprop(self, error):
        self.delta = error

    def update(self, inputs, epsilon, epoch, momentum):
        # There are no weights to update.
        pass


class MaxPoolingLayer(PoolingLayer):

    """
    Max pooling layer.
    """

    def __init__(self, name, backend, batch_size, nfm, ifmshape, pshape,
                 stride):
        super(MaxPoolingLayer, self).__init__(name, backend, batch_size, nfm,
                                              ifmshape, pshape, stride)
        self.maxinds = backend.zeros((batch_size * nfm, self.ofmsize),
                                     dtype='i32')
        self.nout = nfm * self.ofmsize
        self.output = backend.zeros((batch_size, self.nout))

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
        squished_output = self.backend.squish(self.output, self.nfm)
        for dst in range(self.ofmsize):
            # For this output unit, get the corresponding receptive fields
            # within all input feature maps.
            rf = inputs.take(self.links.take(dst, axis=0), axis=1)
            # Save the index of the maximum value within the receptive fields.
            self.maxinds[:, dst] = rf.argmax(axis=1)
            # Set the pre-activations to the maximum value.
            # "fancy" indexing ended up being faster than 1-D take() approach
            maxvals = rf[range(rf.shape[0]), self.maxinds.take(dst, axis=1)]
            squished_output[:, dst] = maxvals

        # Reshape back to original shape.
        self.output = squished_output.reshape((self.output.shape))

    def error(self):
        berror = self.backend.zeros((self.batch_size, self.nin))
        # Reshape the backpropagated error matrix to have one
        # row per feature map.
        rberror = self.backend.squish(berror, self.nfm)
        # Reshape the delta matrix to have one row per feature map.
        rdelta = self.backend.squish(self.delta, self.nfm)
        for dst in range(self.ofmsize):
            links = self.links.take(dst, axis=0)
            colinds = self.maxinds.take(dst, axis=1)
            inds = links.take(colinds, axis=0)
            rberror[range(rberror.shape[0]), inds] += rdelta.take(dst, axis=1)
        berror = rberror.reshape(berror.shape)
        return berror


class L2PoolingLayer(PoolingLayer):

    """
    L2 pooling layer. Each receptive field is pooled to obtain its L2 norm
    as output.
    """

    def __init__(self, name, backend, batch_size, nfm, ifmshape, pshape,
                 stride):
        super(L2PoolingLayer, self).__init__(name, backend, batch_size, nfm,
                                             ifmshape, pshape, stride)
        self.normalized_rf = backend.zeros((batch_size * nfm, self.ifmsize))
        self.nout = nfm * self.ofmsize
        self.output = backend.zeros((batch_size, self.nout))

    def __str__(self):
        return ("L2PoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        squished_inputs = self.backend.squish(inputs, self.nfm)
        squished_output = self.backend.squish(self.output, self.nfm)
        for dst in range(self.ofmsize):
            inds = self.links.take(dst, axis=0)
            rf = squished_inputs.take(inds, axis=1)
            squished_output[:, dst] = rf.norm(axis=1)
            denom = squished_output[:, range(dst, dst + 1)].repeat(self.psize,
                                                                   axis=1)
            # If the L2 norm is zero, the entire receptive field must be zeros.
            # In that case, we set the L2 norm to 1 before using it to
            # normalize the receptive field.
            denom[denom == 0] = 1
            self.normalized_rf[:, inds] = rf / denom
        self.output = squished_output.reshape((self.output.shape))

    def error(self):
        berror = self.backend.zeros((self.batch_size, self.nin))
        rberror = self.backend.squish(berror, self.nfm)
        rdelta = self.backend.squish(self.delta, self.nfm)
        for dst in range(self.ofmsize):
            links = self.links.take(dst, axis=0)
            rberror[:, links] += (self.normalized_rf[:, links] *
                                  rdelta.take(range(dst, dst + 1), axis=1))
        berror = rberror.reshape(berror.shape)
        return berror


class AveragePoolingLayer(PoolingLayer):

    """
    Average pooling.
    """

    def __init__(self, name, backend, batch_size, nfm, ifmshape, pshape,
                 stride):
        super(AveragePoolingLayer, self).__init__(name, backend, batch_size,
                                                  nfm, ifmshape, pshape,
                                                  stride)
        self.nout = nfm * self.ofmsize
        self.output = backend.zeros((batch_size, self.nout))

    def __str__(self):
        return ("AveragePoolingLayer %s: %d nin, %d nout, "
                "utilizing %s backend\n\t" %
                (self.name, self.nin, self.nout,
                 self.backend.__class__.__name__))

    def fprop(self, inputs):
        squished_inputs = self.backend.squish(inputs, self.nfm)
        squished_output = self.backend.squish(self.output, self.nfm)
        for dst in range(self.ofmsize):
            inds = self.links.take(dst, axis=0)
            rf = squished_inputs.take(inds, axis=1)
            squished_output[:, dst] = rf.mean(axis=1)
        self.output = squished_output.reshape((self.output.shape))

    def error(self):
        berror = self.backend.zeros((self.batch_size, self.nin))
        rberror = self.backend.squish(berror, self.nfm)
        rdelta = self.backend.squish(self.delta, self.nfm)
        rdelta /= self.psize
        for dst in range(self.ofmsize):
            links = self.links.take(dst, axis=0)
            rberror[:, links] += rdelta.take(range(dst, dst + 1), axis=1)
        berror = rberror.reshape(berror.shape)
        return berror


class LCNLayer(LocalLayer):

    """
    Local contrast normalization.
    """

    def __init__(self, name, backend, batch_size, nifm, ifmshape, fshape,
                 stride):
        super(LCNLayer, self).__init__(name, backend, batch_size, nifm,
                                       ifmshape, fshape, stride)
        self.nin = nifm * self.ifmsize
        self.nout = nifm * self.ifmsize
        self.output = backend.zeros((batch_size, self.nin))
        self.filter = self.normalized_gaussian_filters(nifm, fshape)
        self.meanfm = self.backend.zeros((self.batch_size,
                                          nifm * self.ofmsize))
        self.ex_meanfm = self.backend.zeros((self.batch_size, self.ifmheight,
                                             self.ifmwidth))
        self.inset_row = self.ifmheight - self.ofmheight
        self.inset_col = self.ifmwidth - self.ofmwidth

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
        for dst in range(self.ofmsize):
            rflinks = self.rlinks[dst]
            prod = self.backend.dot(inputs.take(rflinks, axis=1),
                                    self.filter)
            self.meanfm[:, range(dst, dst + 1)] = prod

        # TODO: handle edges better.
        self.rmeanfm = self.meanfm.reshape((self.batch_size, self.ofmheight,
                                            self.ofmwidth))
        for row in range(self.ex_meanfm.shape[0]):
            self.ex_meanfm[row,
                           self.inset_row:(self.inset_row + self.ofmheight),
                           self.inset_col:(self.inset_col + self.ofmwidth)
                           ] = (self.rmeanfm[row])

        self.rex_meanfm = self.ex_meanfm.reshape((self.batch_size, self.nin))
        res = inputs.copy()
        for i in range(self.nifm):
            res[:, i * self.ifmsize:(i + 1) * self.ifmsize] -= self.rex_meanfm
        return res

    def div_normalize(self, inputs):
        res = inputs.copy()
        res *= res
        denom = self.backend.zeros((self.batch_size, self.nin))
        for dst in range(self.ofmsize):
            rflinks = self.rlinks[dst]
            prod = self.backend.dot(res.take(rflinks, axis=1),
                                    self.filter)
            denom[:, range(dst, dst + 1)] = prod
        denom = self.backend.sqrt(denom, out=denom)
        c = denom.mean()
        denom[denom < c] = c
        return inputs / denom

    def fprop(self, inputs):
        self.output[:] = self.sub_normalize(inputs)
        self.output[:] = self.div_normalize(self.output)

    def update(self, inputs, epsilon, epoch, momentum):
        pass

    def error(self):
        berror = self.backend.zeros((self.batch_size, self.nin))
        rberror = self.backend.squish(berror, self.nifm)
        rdelta = self.backend.squish(self.delta, self.nifm)
        rdelta /= self.fsize
        for dst in range(self.ofmsize):
            links = self.links.take(dst, axis=0)
            rberror[:, links] += rdelta.take(range(dst, dst + 1), axis=1)
        berror = rberror.reshape(berror.shape)
        return berror
