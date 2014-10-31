"""
Numerical gradient checking to validate backprop code.
"""

import logging
import numpy as np

from neon.experiments.experiment import Experiment
from neon.datasets.synthetic import UniformRandom
from neon.backends._numpy import Numpy


logger = logging.getLogger(__name__)


class GradientChecker(Experiment):
    """
    In this `Experiment`, a model is trained on a fake training dataset to
    validate the backprop code within the given model.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def transfer(self, experiment):
        self.model = experiment.model
        self.datasets = experiment.datasets

    def save_state(self):
        for ind in xrange(len(self.trainable_layers)):
            layer = self.model.layers[self.trainable_layers[ind]]
            self.weights[ind][:] = layer.weights

    def load_state(self):
        for ind in xrange(len(self.trainable_layers)):
            layer = self.model.layers[self.trainable_layers[ind]]
            layer.weights[:] = self.weights[ind]

    def check_layer(self, layer, inputs, targets):
        # Check up to this many weights.
        nmax = 10
        updates = layer.updates.raw().ravel()
        weights = layer.weights.raw().ravel()
        grads = np.zeros(weights.shape)
        inds = np.random.choice(np.arange(weights.shape[0]),
                                min(weights.shape[0], nmax),
                                replace=False)
        for ind in inds:
            saved = weights[ind]
            weights[ind] += self.eps
            self.model.fprop(inputs)
            cost1 = self.model.cost.apply_function(
                self.model.backend, self.model.layers[-1].output,
                targets, self.model.temp)

            weights[ind] -= 2 * self.eps
            self.model.fprop(inputs)
            cost2 = self.model.cost.apply_function(
                self.model.backend, self.model.layers[-1].output,
                targets, self.model.temp)

            grads[ind] = ((cost1 - cost2) * self.model.layers[-1].nout *
                          layer.learning_rate / (2 * self.eps))
            weights[ind] = saved

        grads -= updates
        diff = np.linalg.norm(grads[inds])
        if diff < self.eps * 10:
            logger.info('diff %g. layer %s OK.' % (diff, layer.name))
            return True

        logger.error('diff %g. gradient check failed on layer %s.' %
                     (diff, layer.name))
        return False

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """
        if not (hasattr(self.model, 'fprop') and hasattr(self.model, 'bprop')):
            logger.error('Config file not compatible.')
            return

        backend_type = type(self.model.backend)
        if backend_type != Numpy:
            logger.error('%s backend is not supported.' % backend_type)
            return

        self.eps = 1e-6
        self.weights = []
        self.trainable_layers = []
        for ind in xrange(len(self.model.layers)):
            layer = self.model.layers[ind]
            if not (hasattr(layer, 'weights') and hasattr(layer, 'updates')):
                continue
            self.weights.append(layer.weights.copy())
            self.trainable_layers.append(ind)

        if not hasattr(layer, 'datasets'):
            self.datasets[0] = UniformRandom(self.model.batch_size,
                                             self.model.batch_size,
                                             self.model.layers[0].nin,
                                             self.model.layers[-1].nout)
            self.datasets[0].backend = self.model.backend
            self.datasets[0].load()

        inputs = self.datasets[0].get_inputs(train=True)['train']
        targets = self.datasets[0].get_targets(train=True)['train']

        self.model.fprop(inputs)
        self.model.bprop(targets, inputs, 0, self.model.momentum)

        self.save_state()
        self.model.fprop(inputs)
        self.model.bprop(targets, inputs, 0, self.model.momentum)
        self.load_state()

        for ind in self.trainable_layers[::-1]:
            layer = self.model.layers[ind]
            result = self.check_layer(layer, inputs, targets)
            if result is False:
                break
