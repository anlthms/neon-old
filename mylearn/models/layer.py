"""
Generic single neural network layer built to handle data from a particular
backend.
"""

import logging

logger = logging.getLogger(__name__)


class Layer(object):
    """
    Single NNet layer built to handle data from a particular backend
    """
    def __init__(self, name, backend, nin, nout, act_fn, weight_init):
        self.name = name
        self.backend = backend
        self.weights = self.backend.gen_weights((nout, nin), weight_init)
        self.act_fn = getattr(self.backend, act_fn)
        self.act_fn_de = self.backend.get_derivative(self.act_fn)
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
                (self.name, self.nin, self.nout, self.act_fn.__name__,
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
        self.output = self.act_fn(self.pre_act)

    def bprop(self, error):
        self.delta = error * self.act_fn_de(self.pre_act)

    def update(self, inputs, epsilon, epoch, momentum):
        inputs = self.backend.append_bias(inputs)
        momentum_coef = self.backend.get_momentum_coef(epoch, momentum)
        self.velocity = (momentum_coef * self.velocity -
                         epsilon * self.backend.dot(self.delta.T(), inputs))
        self.weights += self.velocity

    def error(self):
        return self.backend.dot(self.delta,
                                self.weights.get_slice(0,
                                                       self.weights.shape[1] -
                                                       1,
                                                       axis=1))


class LayerWithNoBias(Layer):
    """
    Single NNet layer with no bias node - temporary code for testing purposes.
    """

    def fprop(self, inputs):
        self.pre_act = self.backend.dot(inputs, self.weights.T())
        self.output = self.act_fn(self.pre_act)

    def update(self, inputs, epsilon, epoch, momentum):
        self.weights -= epsilon * self.backend.dot(self.delta.T(), inputs)

    def error(self):
        return self.backend.dot(self.delta, self.weights)


class AELayer(object):
    """
    Single NNet layer built to handle data from a particular backend used
    in an Autoencoder.
    TODO: merge with generic Layer above.
    """
    def __init__(self, name, backend, nin, nout, act_fn, weight_init,
                 weights=None):
        self.name = name
        self.backend = backend
        if weights is None:
            self.weights = self.backend.gen_weights((nout, nin), weight_init)
        else:
            self.weights = weights
        self.act_fn = getattr(self.backend, act_fn)
        self.act_fn_de = self.backend.get_derivative(self.act_fn)
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
                (self.name, self.nin, self.nout, self.act_fn.__name__,
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
        if self.act_fn == self.backend.noact:
            self.output = self.pre_act
        else:
            self.output = self.act_fn(self.pre_act)

    def bprop(self, error):
        if self.act_fn_de == self.backend.noact_prime:
            self.delta = error
        else:
            self.delta = error * self.act_fn_de(self.pre_act)

    def update(self, inputs, epsilon, epoch):
        self.weights -= epsilon * self.backend.dot(self.delta.T(), inputs)

    def error(self):
        return self.backend.dot(self.delta, self.weights)


class ConvLayer(object):
    """
    Convolutional layer.
    A stride of 1 is assumed.
    """

    def __init__(self, name, backend, batch_size, nifm, ifmshape, fshape,
                 nfilt, weight_init):
        self.name = name
        self.backend = backend
        self.ifmheight, self.ifmwidth = ifmshape
        self.fheight, self.fwidth = fshape
        self.batch_size = batch_size

        ofmheight = self.ifmheight - self.fheight + 1
        ofmwidth = self.ifmwidth - self.fwidth + 1
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.ofmsize = ofmheight * ofmwidth
        self.nout = self.ofmsize * nfilt

        self.nifm = nifm
        self.nfilt = nfilt
        self.fsize = nifm * self.fheight * self.fwidth
        self.weights = self.backend.gen_weights((nfilt, self.fsize),
                                                weight_init)
        self.output = backend.zeros((batch_size, self.nout))
        ofmstarts = self.backend.array(range(0, (self.ofmsize * nfilt),
                                       self.ofmsize))
        # Figure out the connections with the previous layer.
        self.links = backend.zeros((self.ofmsize, self.fsize), dtype='i32')
        ofmlocs = backend.zeros((self.ofmsize, nfilt), dtype='i32')
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

            if (src % self.ifmwidth + self.fwidth) < self.ifmwidth:
                # Slide the filter by 1 cell.
                src += 1
            else:
                # We hit the right edge of the input image.
                # Sweep the filter over to the next row.
                src += self.fwidth
            self.links.set(backend.array(colinds), dst, axis=0)
            ofmlocs.set_slice(ofmstarts + dst, dst, dst + 1, axis=0)
        self.rlinks = self.links.raw()
        self.rofmlocs = ofmlocs.raw()

    def fprop(self, inputs):
        for dst in range(self.ofmsize):
            # Compute the weighted average of the receptive field
            # and store the result within the destination feature map.
            # Do this for all filters in one shot.
            rflinks = self.rlinks[dst]
            prod = self.backend.dot(inputs.get(rflinks, axis=1),
                                    self.weights.T())
            self.output.set(prod, self.rofmlocs[dst], axis=1)

    def bprop(self, error):
        self.delta = error

    def update(self, inputs, epsilon, epoch, momentum):
        wsums = self.backend.zeros(self.weights.shape)
        for dst in range(self.ofmsize):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = self.rlinks[dst]
            delta_slice = self.delta.get(self.rofmlocs[dst], axis=1)
            prod = self.backend.dot(delta_slice.T(), inputs.get(rflinks,
                                                                axis=1))
            wsums.add(prod)
        # Update the filters after summing the weight updates.
        self.weights.sub(epsilon * wsums)

    def error(self):
        berror = self.backend.zeros((self.batch_size,
                                    self.ifmheight * self.ifmwidth *
                                    self.nifm))
        for dst in range(self.ofmsize):
            res = self.backend.dot(self.delta.get(self.rofmlocs[dst], axis=1),
                                   self.weights)
            rflinks = self.rlinks[dst]
            res.add(berror.get(rflinks, axis=1))
            berror.set(res, rflinks, axis=1)
        return berror


class MaxPoolingLayer:
    """
    Max pooling layer.
    The code assumes that there is no overlap between pooling regions.
    """
    def __init__(self, name, backend, batch_size, nfm, ifmshape, pshape,
                 weight_init):
        self.name = name
        self.backend = backend
        self.nfm = nfm
        self.ifmheight, self.ifmwidth = ifmshape
        self.ifmsize = self.ifmheight * self.ifmwidth
        self.pheight, self.pwidth = pshape
        self.psize = self.pheight * self.pwidth
        self.batch_size = batch_size
        assert self.ifmheight % self.pheight == 0
        assert self.ifmwidth % self.pwidth == 0

        self.ofmsize = self.ifmsize / self.psize
        self.nin = nfm * self.ifmsize
        self.nout = nfm * self.ofmsize
        self.output = backend.zeros((batch_size, self.nout))
        self.maxinds = backend.zeros((batch_size * nfm, self.ofmsize),
                                     dtype='i32')
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
            src += self.pwidth
            if (src % self.ifmwidth) == 0:
                # Shift the pooling window down by 1 receptive field.
                src += (self.pheight - 1) * self.ifmwidth
            self.links.set(backend.array(colinds), dst, axis=0)

    def fprop(self, inputs):
        # Reshape the input so that we have a separate row
        # for each input feature map (this is to avoid a loop over
        # each feature map).
        inputs = self.backend.squish(inputs, self.nfm)
        squished_output = self.backend.squish(self.output, self.nfm)
        for dst in range(self.ofmsize):
            # For this output unit, get the corresponding receptive fields
            # within all input feature maps.
            rf = inputs.get(self.links.get(dst, axis=0), axis=1)
            # Save the index of the maximum value within the receptive fields.
            self.maxinds.set(rf.argmax(axis=1), dst, axis=1)
            # Set the pre-activations to the maximum value.
            maxvals = rf.get_elems(self.maxinds.get(dst, axis=1), axis=0)
            squished_output.set(maxvals, dst, axis=1)

        # Reshape back to original shape.
        self.output = squished_output.reshape((self.output.shape))

    def bprop(self, error):
        self.delta = error

    def update(self, inputs, epsilon, epoch, momentum):
        # There are no weights to update.
        pass

    def error(self):
        berror = self.backend.zeros((self.batch_size, self.nin))
        # Reshape the backpropagated error matrix to have one
        # row per feature map.
        rberror = self.backend.squish(berror, self.nfm)
        # Reshape the delta matrix to have one row per feature map.
        rdelta = self.backend.squish(self.delta, self.nfm)
        for dst in range(self.ofmsize):
            links = self.links.get(dst, axis=0)
            colinds = self.maxinds.get(dst, axis=1)
            inds = links.get(colinds, axis=0)
            rberror.set_elems(rdelta.get(dst, axis=1),
                              inds, axis=0)
        berror = rberror.reshape(berror.shape)
        return berror
