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
        self.velocity = self.backend.Tensor(0.0)
        self.y = None
        self.z = None

    def __str__(self):
        return ("Layer %s: %d inputs, %d nodes, %s act_fn, "
                "utilizing %s backend\n\t"
                "y: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "z: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t"
                "velocity: mean=%.05f, min=%.05f, max=%.05f" %
                (self.name, self.nin, self.nout, self.act_fn.__name__,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.y),
                 self.backend.min(self.y),
                 self.backend.max(self.y),
                 self.backend.mean(self.z),
                 self.backend.min(self.z),
                 self.backend.max(self.z),
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights),
                 self.backend.mean(self.velocity),
                 self.backend.min(self.velocity),
                 self.backend.max(self.velocity)))

    def fprop(self, inputs):
        inputs = self.backend.append_bias(inputs)
        self.z = self.backend.dot(inputs, self.weights.T())
        self.y = self.act_fn(self.z)
        return self.y

    def bprop(self, error):
        self.delta = error * self.act_fn_de(self.z)

    def update(self, inputs, epsilon, epoch, momentum):
        inputs = self.backend.append_bias(inputs)
        momentum_coef = self.backend.get_momentum_coef(epoch, momentum)
        self.velocity = (momentum_coef * self.velocity -
                         epsilon * self.backend.dot(self.delta.T(), inputs))
        self.weights += self.velocity

    def error(self):
        return self.backend.dot(self.delta, self.weights[:, :-1])


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
        self.y = None
        self.z = None

    def __str__(self):
        return ("Layer %s: %d inputs, %d nodes, %s act_fn, "
                "utilizing %s backend\n\t"
                "y: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "z: mean=%.05f, min=%.05f, max=%.05f,\n\t"
                "weights: mean=%.05f, min=%.05f, max=%.05f\n\t" %
                (self.name, self.nin, self.nout, self.act_fn.__name__,
                 self.backend.__class__.__name__,
                 self.backend.mean(self.y),
                 self.backend.min(self.y),
                 self.backend.max(self.y),
                 self.backend.mean(self.z),
                 self.backend.min(self.z),
                 self.backend.max(self.z),
                 self.backend.mean(self.weights),
                 self.backend.min(self.weights),
                 self.backend.max(self.weights)))

    def fprop(self, inputs):
        self.z = self.backend.dot(inputs, self.weights.T())
        if self.act_fn == self.backend.noact:
            self.y = self.z
        else:
            self.y = self.act_fn(self.z)
        return self.y

    def bprop(self, error):
        if self.act_fn_de == self.backend.noact_prime:
            self.delta = error
        else:
            self.delta = error * self.act_fn_de(self.z)

    def update(self, inputs, epsilon, epoch):
        self.weights -= epsilon * self.backend.dot(self.delta.T(), inputs)

    def error(self):
        return self.backend.dot(self.delta, self.weights)
