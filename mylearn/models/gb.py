"""
Contains code to train Google Brain models and run inference.
"""

import logging
import math
import os

from mylearn.models.layer import LocalFilteringLayer
from mylearn.models.mlp import MLP
from mylearn.util.persist import ensure_dirs_exist

logger = logging.getLogger(__name__)


class GB(MLP):
    """
    Google Brain class
    """

    def pretrain(self, inputs):
        logger.info('commencing unsupervised pretraining')
        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        for ind in range(len(self.trainable_layers)):
            layer = self.layers[self.trainable_layers[ind]]
            pooling = self.layers[self.trainable_layers[ind] + 1]
            layer.pretrain_mode(pooling)
            for epoch in xrange(self.num_pretrain_epochs):
                error = 0.0
                for batch in xrange(num_batches):
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                    output = inputs[start_idx:end_idx]
                    # Forward propagate the input all the way to
                    # the layer that we are pretraining.
                    for i in xrange(self.trainable_layers[ind]):
                        self.layers[i].fprop(output)
                        output = self.layers[i].output
                    error += layer.pretrain(output, self.pretrain_cost, epoch,
                                            self.momentum)
                logger.info('epoch: %d, total training error: %0.5f' %
                            (epoch, error / num_batches))
                self.save_figs([output, layer.defilter.output],
                               [os.path.join('recon', 'input'),
                                os.path.join('recon', 'output')], ind)
        # Switch the layers from pretraining to training mode.
        for layer in self.layers:
            if isinstance(layer, LocalFilteringLayer):
                layer.train_mode()
        if self.num_pretrain_epochs > 0:
            self.visualize()

    def train(self, inputs, targets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing supervised training')
        tempbuf = self.backend.zeros((self.batch_size, targets.shape[1]))
        self.temp = [tempbuf, tempbuf.copy()]

        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                self.fprop(inputs[start_idx:end_idx])
                if epoch <= self.num_epochs / 10:
                    self.bprop_last(targets[start_idx:end_idx],
                                    inputs[start_idx:end_idx],
                                    epoch, self.momentum)
                else:
                    self.bprop(targets[start_idx:end_idx],
                               inputs[start_idx:end_idx],
                               epoch, self.momentum)
                error += self.cost.apply_function(self.backend,
                                                  self.layers[-1].output,
                                                  targets[start_idx:end_idx],
                                                  self.temp)
            logger.info('epoch: %d, total training error: %0.5f' %
                        (epoch, error / num_batches))

    def bprop_last(self, targets, inputs, epoch, momentum):
        # Backprop on just the last layer.
        error = self.cost.apply_derivative(self.backend,
                                           self.layers[-1].output, targets,
                                           self.temp)
        self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                            out=error)
        self.layers[-1].bprop(error, self.layers[-2].output, epoch, momentum)

    def fit(self, datasets):
        inputs = datasets[0].get_inputs(train=True)['train']
        self.nrecs, self.nin = inputs.shape
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = self.nrecs
        self.trainable_layers = []
        for ind in xrange(self.nlayers):
            layer = self.layers[ind]
            if isinstance(layer, LocalFilteringLayer):
                self.trainable_layers.append(ind)
            logger.info('created layer:\n\t%s' % str(layer))

        self.pretrain(inputs)
        targets = datasets[0].get_targets(train=True)['train']
        self.train(inputs, targets)

    def normalize(self, data):
        norms = data.norm(axis=1)
        self.backend.divide(data, norms.reshape((norms.shape[0], 1)),
                            out=data)

    def visualize(self):
        """
        This function tries to generate synthetic input data that maximizes
        the probability of activating the output neurons.
        """
        import matplotlib.pyplot as plt
        logger.info('visualize')
        inputs = self.backend.uniform(low=-0.1, high=0.1,
                                      size=(self.batch_size, self.nin))
        self.normalize(inputs)
        lastlayer = self.layers[-2]
        self.fprop(inputs)
        outmax = lastlayer.output[range(self.batch_size),
                                  range(self.batch_size)]
        ifmshape = (self.layers[0].ifmheight, self.layers[0].ifmwidth)
        inc = 0.1
        # Do a greedy search to find input data that maximizes the output
        # of neurons in the last LCN layer.
        for loops in range(20):
            inc *= -0.9
            count = 0
            for col in range(self.nin):
                saved = inputs.copy()
                inputs[:, col] += inc
                self.normalize(inputs)
                self.fprop(inputs)
                output = lastlayer.output[range(self.batch_size),
                                          range(self.batch_size)]
                maxinds = output > outmax
                notinds = output < outmax
                outmax[maxinds] = output[maxinds]
                inputs[notinds, :] = saved[notinds, :]
                count += maxinds.sum()
            logger.info('loop %d inc %.4f count %d' % (loops, inc, count))
            for ind in range(self.batch_size):
                if self.layers[0].nifm == 3:
                    img = inputs[ind].raw().reshape((3, ifmshape[0],
                                                     ifmshape[1]))
                    rimg = img.copy().reshape((ifmshape[0], ifmshape[1], 3))
                    for dim in range(3):
                        rimg[:ifmshape[0], :ifmshape[1], dim] = (
                            img[dim, :ifmshape[0], :ifmshape[1]])
                else:
                    assert self.layers[0].nifm == 1
                    rimg = inputs[ind].raw().reshape(ifmshape)
                plt.imshow(rimg, interpolation='nearest', cmap='gray')
                plt.savefig(ensure_dirs_exist(os.path.join('imgs', 'img') +
                                              str(ind)))

    def save_figs(self, imgs, names, ind):
        import matplotlib.pyplot as plt
        assert len(names) == len(imgs)
        for i in range(len(names)):
            img = imgs[i].raw()[0]
            width = math.sqrt(img.shape[0])
            if width * width != img.shape[0]:
                width = math.sqrt(img.shape[0] / 3)
                img = img.reshape((3, width, width))
                rimg = img.copy().reshape((width, width, 3))
                for dim in range(3):
                    rimg[:width, :width, dim] = img[dim, :width, :width]
                plt.imshow(rimg, interpolation='nearest')
            else:
                plt.imshow(img.reshape((width, width)),
                           interpolation='nearest', cmap='gray')
            plt.savefig(ensure_dirs_exist(names[i] + str(ind)))
