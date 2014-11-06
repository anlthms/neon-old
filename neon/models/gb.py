"""
Contains code to train Google Brain models and run inference.
"""

import logging
import math
import os

from neon.models.layer import LocalFilteringLayer
from neon.models.mlp import MLP
from neon.util.persist import ensure_dirs_exist
import time

logger = logging.getLogger(__name__)


class GB(MLP):

    """
    Google Brain class
    """

    def pretrain(self, inputs):
        start_time = time.time()
        logger.debug('commencing unsupervised pretraining')
        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        for ind in range(len(self.trainable_layers)):
            layer = self.layers[self.trainable_layers[ind]]
            pooling = self.layers[self.trainable_layers[ind] + 1]
            layer.pretrain_mode(pooling)
            for epoch in xrange(self.num_pretrain_epochs):
                tcost = 0.0
                trcost = 0.0
                tspcost = 0.0
                for batch in xrange(num_batches):
                    logger.info('batch = %d' % (batch))
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                    output = inputs[start_idx:end_idx]
                    # Forward propagate the input all the way to
                    # the layer that we are pretraining.
                    for i in xrange(self.trainable_layers[ind]):
                        self.layers[i].fprop(output)
                        output = self.layers[i].output
                    rcost, spcost = layer.pretrain(output,
                                                   self.pretrain_cost,
                                                   epoch)
                    trcost += rcost
                    tspcost += spcost
                tcost = trcost + tspcost
                logger.info('layer: %d, epoch: %d, cost: %0.2f + %0.2f ='
                            ' %0.2f' % (self.trainable_layers[ind], epoch,
                                        trcost / num_batches, tspcost /
                                        num_batches,
                                        tcost / num_batches))
                if self.visualize:
                    self.save_figs(layer.nifm, layer.ifmshape,
                                   [output, layer.defilter.output],
                                   [os.path.join('recon', 'input'),
                                    os.path.join('recon', 'output')], ind)
        end_time = time.time()
        logger.info('Time taken: %0.2f' % (end_time - start_time))

        # Switch the layers from pretraining to training mode.
        for layer in self.layers:
            if isinstance(layer, LocalFilteringLayer):
                layer.train_mode()

    def train(self, inputs, targets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing supervised training')
        tempbuf = self.backend.zeros((self.batch_size, targets.shape[1]))
        self.temp = [tempbuf, tempbuf.copy()]
        start_time = time.time()
        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):
                logger.debug('batch = %d' % (batch))
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                self.fprop(inputs[start_idx:end_idx])
                if epoch < self.num_initial_epochs:
                    self.bprop_last(targets[start_idx:end_idx],
                                    inputs[start_idx:end_idx],
                                    epoch)
                else:
                    self.bprop(targets[start_idx:end_idx],
                               inputs[start_idx:end_idx],
                               epoch)
                error += self.cost.apply_function(self.backend,
                                                  self.layers[-1].output,
                                                  targets[start_idx:end_idx],
                                                  self.temp)
            logger.info('epoch: %d, training error: %0.5f' %
                        (epoch, error / num_batches))
        end_time = time.time()
        logger.info('Time taken: %0.2f' % (end_time - start_time))

    def check_node_predictions(self, inputs, targets, node, cls):
        """
        Spot-check the classification accuracy of an output neuron
        for the given class.
        """
        from sklearn import metrics
        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        labels = self.backend.zeros((targets.shape[0]), dtype=int)
        labels[targets[:, cls] == 0] = 0
        labels[targets[:, cls] == 1] = 1
        auc = 0.0
        for batch in xrange(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, self.nrecs)
            self.fprop(inputs[start_idx:end_idx])
            # Get the output of the last LCN layer.
            pred = self.layers[-2].output[:, node]
            auc += metrics.roc_auc_score(
                labels[start_idx:end_idx].raw(), pred.raw())
        auc /= num_batches
        return auc

    def check_predictions(self, inputs, targets, test_inputs, test_targets):
        """
        Check the classification accuracy of output neurons.
        """
        from sklearn import metrics
        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        labels = self.backend.zeros((targets.shape[0]), dtype=int)
        sum = 0.0
        for cls in xrange(targets.shape[1]):
            labels[targets[:, cls] == 0] = 0
            labels[targets[:, cls] == 1] = 1
            auc = self.backend.zeros((self.layers[-2].output.shape[1]))
            for batch in xrange(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                self.fprop(inputs[start_idx:end_idx])
                # Get the output of the last LCN layer.
                for node in xrange(auc.shape[0]):
                    pred = self.layers[-2].output[:, node]
                    auc[node] += metrics.roc_auc_score(
                        labels[start_idx:end_idx].raw(), pred.raw())
            auc /= num_batches
            maxnode = self.backend.argmax(auc).raw()
            maxauc = auc[maxnode]
            # Check classification accuracy of the best neuron on the test set.
            testauc = self.check_node_predictions(test_inputs, test_targets,
                                                  maxnode, cls)
            logger.info(
                'class: %d best node: %d train auc: %.4f test auc: %.4f' %
                (cls, maxnode, maxauc, testauc))
            sum += maxauc
        logger.info('average max auc %.4f' % (sum / targets.shape[1]))

    def bprop_last(self, targets, inputs, epoch):
        # Backprop on just the last layer.
        error = self.cost.apply_derivative(self.backend,
                                           self.layers[-1].output, targets,
                                           self.temp)
        self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                            out=error)
        self.layers[-1].bprop(error, self.layers[-2].output, epoch)

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

        targets = datasets[0].get_targets(train=True)['train']
        if self.pretraining:
            self.pretrain(inputs)
            if self.visualize:
                self.compute_optimal_stimulus()
        if self.spot_check:
            test_inputs = datasets[0].get_inputs(test=True)['test']
            test_targets = datasets[0].get_targets(test=True)['test']
            self.check_predictions(inputs, targets, test_inputs, test_targets)
        if self.num_epochs > 0:
            self.train(inputs, targets)

    def normalize(self, data):
        norms = data.norm(axis=1)
        self.backend.divide(data, norms.reshape((norms.shape[0], 1)),
                            out=data)

    def compute_optimal_stimulus(self):
        """
        This function tries to generate synthetic input data that maximizes
        the probability of activating the output neurons.
        """
        import matplotlib.pyplot as plt
        logger.info('visualizing features...')
        inputs = self.backend.ones((self.batch_size, self.nin))
        self.normalize(inputs)
        lastlayer = self.layers[-2]
        self.fprop(inputs)
        outmax = lastlayer.output[range(self.batch_size),
                                  range(self.batch_size)]
        ifmshape = (self.layers[0].ifmheight, self.layers[0].ifmwidth)
        inc = 0.1
        # Do a greedy search to find input data that maximizes the output
        # of neurons in the last LCN layer.
        for loops in range(10):
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
                    plt.imshow(rimg, interpolation='nearest')
                else:
                    assert self.layers[0].nifm == 1
                    rimg = inputs[ind].raw().reshape(ifmshape)
                    plt.imshow(rimg, interpolation='nearest', cmap='gray')
                plt.savefig(ensure_dirs_exist(os.path.join('imgs', 'img') +
                                              str(ind)))

    def save_figs(self, nfm, fmshape, imgs, names, ind):
        import matplotlib.pyplot as plt
        assert len(names) == len(imgs)
        height, width = fmshape
        for i in range(len(names)):
            img = imgs[i].raw()[0]
            img = img.reshape((nfm, height, width))
            if nfm == 3:
                # Plot in color.
                rimg = img.copy().reshape((height, width, 3))
                for dim in range(3):
                    rimg[:height, :width, dim] = img[dim, :height, :width]
                plt.imshow(rimg, interpolation='nearest')
            else:
                # Save the first feature map.
                plt.imshow(img[0].reshape((height, width)),
                           interpolation='nearest', cmap='gray')
            plt.savefig(ensure_dirs_exist(names[i] + str(ind)))
