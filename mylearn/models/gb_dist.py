"""
Contains code to train Google Brain models and run inference.
"""

import logging
import math
import os

from mylearn.models.mlp import MLPDist
from mylearn.models.layer_dist import LocalFilteringLayerDist
from mylearn.util.persist import ensure_dirs_exist
from mylearn.util.distarray.global_array import GlobalArray
from mpi4py import MPI
import time

logger = logging.getLogger(__name__)


class GBDist(MLPDist):

    """
    MPI Distributed Google Brain class
    """

    def fit(self, datasets):
        inputs = datasets[0].get_inputs(train=True)['train']
        self.nrecs, self.nin = inputs.shape
        self.nlayers = len(self.layers)
        if 'batch_size' not in self.__dict__:
            self.batch_size = self.nrecs
        self.trainable_layers = []
        for ind in xrange(self.nlayers):
            layer = self.layers[ind]
            if isinstance(layer, LocalFilteringLayerDist):
                self.trainable_layers.append(ind)
            # logger.info('created layer:\n\t%s' % str(layer))

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

    def pretrain(self, inputs):
        start_time = time.time()
        logger.info('commencing unsupervised pretraining')
        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        ccomm = None  # for supervised this will be a global or returned value
        self.inputs_dist = dict()
        for ind in range(len(self.trainable_layers)):
            layer = self.layers[self.trainable_layers[ind]]
            # MPI: initialize the distributed global array
            # this call assumes that filters are square
            if ind == 0:
                create_comm = True
                act_size_height = layer.ifmshape[0]
                act_size_width = layer.ifmshape[1]
            else:
                create_comm = False
                # assuming LCN layer doesn't reduce size of image
                act_size_height = self.inputs_dist[
                    self.trainable_layers[ind] - 1].local_array.height
                act_size_width = self.inputs_dist[
                    self.trainable_layers[ind] - 1].local_array.width

            # GlobalArray for local filtering layer
            self.inputs_dist[self.trainable_layers[ind]] = \
                GlobalArray(batch_size=self.batch_size,
                            act_size_height=layer.ifmshape[0],
                            act_size_width=layer.ifmshape[1],
                            act_channels=layer.nifm,
                            filter_size=layer.fwidth,
                            backend=self.backend,
                            create_comm=create_comm,
                            ccomm=ccomm,
                            h=act_size_height,
                            w=act_size_width)
            if create_comm:
                ccomm = self.inputs_dist[self.trainable_layers[ind]].ccomm
                create_comm = False

            # update params to account for halos
            layer.adjust_for_halos([self.inputs_dist[
                self.trainable_layers[ind]].local_array.height_with_halos,
                self.inputs_dist[self.trainable_layers[
                    ind]].local_array.width_with_halos],
                self.inputs_dist[self.trainable_layers[
                                 ind]].local_array.top_left_row_output,
                self.inputs_dist[self.trainable_layers[
                                 ind]].local_array.top_left_col_output)

            pooling = self.layers[self.trainable_layers[ind] + 1]

            # todo: will be a function of prev layer rather than input
            # assumes stride of 1 for pooling layer
            self.inputs_dist[self.trainable_layers[ind] + 1] = \
                GlobalArray(batch_size=self.batch_size,
                            act_size_height=pooling.ifmheight,
                            act_size_width=pooling.ifmwidth,
                            act_channels=pooling.nfm,
                            filter_size=pooling.pwidth,
                            backend=self.backend,
                            create_comm=create_comm,
                            ccomm=ccomm,
                            h=layer.ifmshape[0] - layer.fheight + 1,
                            w=layer.ifmshape[1] - layer.fwidth + 1)
            pooling.adjust_for_dist([self.inputs_dist[
                self.trainable_layers[ind] + 1].local_array.height_with_halos,
                self.inputs_dist[self.trainable_layers[
                    ind] + 1].local_array.width_with_halos])
            layer.pretrain_mode(pooling)

            # temp1 stores a temp buffer without the chunk
            layer.defilter.temp1 = [self.backend.zeros(
                (self.batch_size, self.inputs_dist[
                    self.trainable_layers[ind]].local_array.local_array_size))]

            lcn = self.layers[self.trainable_layers[ind] + 2]
            self.inputs_dist[self.trainable_layers[ind] + 2] = \
                GlobalArray(batch_size=self.batch_size,
                            act_size_height=lcn.ifmheight,
                            act_size_width=lcn.ifmwidth,
                            act_channels=lcn.nfm,
                            filter_size=lcn.fwidth,
                            backend=self.backend,
                            create_comm=create_comm,
                            ccomm=ccomm,
                            h=pooling.ifmheight -
                            pooling.pheight + 1,
                            w=pooling.ifmwidth - pooling.pwidth + 1,
                            lcn_layer_flag=True)
            self.lcn_global_size = lcn.ifmheight * lcn.ifmwidth
            self.lcn_global_width = lcn.ifmwidth
            lcn.adjust_for_dist([self.inputs_dist[
                self.trainable_layers[ind] + 2].local_array.height_with_halos,
                self.inputs_dist[self.trainable_layers[
                    ind] + 2].local_array.width_with_halos],
                border_id=self.inputs_dist[
                    self.trainable_layers[ind]].border_id,
                output_height=pooling.ifmheight - pooling.pheight + 1,
                output_width=pooling.ifmwidth - pooling.pwidth + 1,
                inputs_dist=self.inputs_dist[self.trainable_layers[ind] + 2])

            for epoch in xrange(self.num_pretrain_epochs):
                tcost = 0.0
                trcost = 0.0
                tspcost = 0.0
                trcost_sum = 0.0
                tspcost_sum = 0.0
                for batch in xrange(num_batches):  # num_batches
                    if MPI.COMM_WORLD.rank == 0:
                        print 'batch =', batch
                    start_idx = batch * self.batch_size
                    end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                    output = inputs[start_idx:end_idx]
                    # Forward propagate the input all the way to
                    # the layer that we are pretraining.
                    for i in xrange(self.trainable_layers[ind]):
                        # do MPI exchanges for LocalFilteringDist layers
                        # MPI: set mini-batch to local_image
                        self.inputs_dist[i].local_array.local_image = output
                        # perform halo exchanges
                        self.inputs_dist[i].local_array.send_recv_halos()
                        # make consistent chunk
                        self.inputs_dist[
                            i].local_array.make_local_chunk_consistent()

                        self.layers[i].fprop(
                            self.inputs_dist[i].local_array.chunk)

                        output = self.layers[i].output

                    # MPI: set mini-batch to local_image
                    self.inputs_dist[self.trainable_layers[
                        ind]].local_array.local_image = output
                    # perform halo exchanges
                    if ind == 1:
                        self.inputs_dist[self.trainable_layers[
                            ind]].local_array.send_recv_halos(True)
                    else:
                        self.inputs_dist[self.trainable_layers[
                            ind]].local_array.send_recv_halos()
                    # make consistent chunk
                    self.inputs_dist[self.trainable_layers[
                        ind]].local_array.make_local_chunk_consistent()

                    rcost, spcost = layer.pretrain(self.inputs_dist,
                                                   self.trainable_layers[ind],
                                                   self.pretrain_cost,
                                                   epoch,
                                                   self.momentum)

                    trcost += rcost
                    tspcost += spcost

                # accumulate trcost and tspcost cost across all nodes
                trcost_sum = MPI.COMM_WORLD.reduce(trcost,
                                                   op=MPI.SUM, root=0)
                tspcost_sum = MPI.COMM_WORLD.reduce(tspcost,
                                                    op=MPI.SUM, root=0)
                if MPI.COMM_WORLD.rank == 0:
                    tcost = trcost_sum + tspcost_sum
                    logger.info('epoch: %d, cost: %0.2f + %0.2f = %0.2f' %
                                (epoch, trcost_sum / num_batches,
                                 tspcost_sum / num_batches,
                                 tcost / num_batches))
                if self.visualize:
                    self.save_figs(layer.nifm, layer.ifmshape,
                                   [output, layer.defilter.output],
                                   [os.path.join('recon', 'input'),
                                    os.path.join('recon', 'output')], ind)
        print "Done with pretraining"
        end_time = time.time()
        print MPI.COMM_WORLD.rank, 'time taken: ', end_time - start_time
        # Switch the layers from pretraining to training mode.
        for layer in self.layers:
            if isinstance(layer, LocalFilteringLayerDist):
                layer.train_mode()

    def train(self, inputs, targets):
        """
        Learn model weights on the given datasets.
        """
        logger.info('commencing supervised training')
        tempbuf = self.backend.zeros((self.batch_size, targets.shape[1]))
        self.temp = [tempbuf, tempbuf.copy()]
        start_time = time.time()

        top_lcn_layer_index = len(self.inputs_dist) - 1
        lcn_tl_row_output = self.inputs_dist[
            top_lcn_layer_index].local_array.top_left_row_output
        lcn_tl_col_output = self.inputs_dist[
            top_lcn_layer_index].local_array.top_left_col_output
        #fully connected layer
        self.layers[-1].adjust_for_dist(self.layers[-2].nout,
                                        self.layers[
                                            -3].ofmshape, self.layers[-2].nfm,
                                        self.lcn_global_size,
                                        self.lcn_global_width,
                                        lcn_tl_row_output,
                                        lcn_tl_col_output)

        self.agg_output = self.backend.zeros(
            self.layers[-1].output.shape, 'float32')

        num_batches = int(math.ceil((self.nrecs + 0.0) / self.batch_size))
        for epoch in xrange(self.num_epochs):
            error = 0.0
            for batch in xrange(num_batches):  # num_batches
                if MPI.COMM_WORLD.rank == 0:
                    print 'batch =', batch
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, self.nrecs)
                self.fprop(inputs[start_idx:end_idx], self.inputs_dist)

                if epoch < self.num_initial_epochs:
                    #only bprop on FC layers
                    self.bprop_last(targets[start_idx:end_idx],
                                    inputs[start_idx:end_idx],
                                    epoch, self.momentum)
                else:
                    #bprop through full stack
                    self.bprop(targets[start_idx:end_idx],
                               inputs[start_idx:end_idx],
                               epoch, self.momentum)
                if MPI.COMM_WORLD.rank == 0:
                    error += self.cost.apply_function(self.backend,
                                                      self.layers[-1].output,
                                                      targets[
                                                          start_idx:end_idx],
                                                      self.temp)
            if MPI.COMM_WORLD.rank == 0:
                logger.info('epoch: %d, training error: %0.5f' %
                            (epoch, error / num_batches))
        end_time = time.time()
        print MPI.COMM_WORLD.rank, 'time taken: ', end_time - start_time

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

    def bprop_last(self, targets, inputs, epoch, momentum):
        # Backprop on just the last layer.
        error = self.backend.zeros((self.batch_size, self.layers[-1].nout))
        if MPI.COMM_WORLD.rank == 0:
            error = self.cost.apply_derivative(self.backend,
                                               self.layers[-1].output, targets,
                                               self.temp)
            self.backend.divide(error, self.backend.wrap(targets.shape[0]),
                                out=error)
        else:
            error = self.backend.zeros((self.batch_size, self.layers[-1].nout))
        # broadcast the error matrix
        error._tensor = MPI.COMM_WORLD.bcast(error.raw())

        self.layers[-1].bprop(error, self.layers[-2].output, epoch, momentum)

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
