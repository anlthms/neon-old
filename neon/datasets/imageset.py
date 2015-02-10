# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
ImageNet 1k dataset
More information at: http://www.image-net.org/download-imageurls
Sign up for an ImageNet account to download the dataset!
"""

import logging
import numpy as np
import os
import cPickle
from PIL import Image
from StringIO import StringIO
from time import time
from neon.datasets.dataset import Dataset
from neon.backends.gpu import GPU, GPUTensor
from neon.util.compat import range
from neon.util.param import opt_param, req_param
import sys
import threading
import Queue
import multiprocessing as mp
# importing scipy.io breaks multiprocessing! don't do it here!

logger = logging.getLogger(__name__)

# global queues to start threads
macroq = Queue.Queue()
miniq = Queue.Queue()
gpuq = Queue.Queue()
macroq_flag = False
miniq_flag = False
gpuq_flag = False


def my_pickle(filename, data):
    with open(filename, "w") as fo:
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)


def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents


class LoadFile(threading.Thread):

    '''
    thread that handles loading the macrobatch from pickled file on disk
    '''

    def __init__(self, file_name_queue, macro_batch_queue):
        threading.Thread.__init__(self)
        # queue with file names of macro batches
        self.file_name_queue = file_name_queue
        # queue with file contents
        self.macro_batch_queue = macro_batch_queue

    def run(self):
        fname = self.file_name_queue.get(block=True)
        self.macro_batch_queue.put(my_unpickle(fname))
        self.file_name_queue.task_done()
        if not macroq.empty():
            t = macroq.get()
            t.start()
            macroq.task_done()
        else:
            global macroq_flag
            macroq_flag = False


class DecompressImages(threading.Thread):

    '''
    thread that decompresses/translates/crops/flips jpeg images on CPU in mini-
    batches
    '''

    def __init__(self, mb_id, mini_batch_queue, batch_size, output_image_size,
                 cropped_image_size, macro_data, backend,
                 num_processes, mean_img, predict, dotransforms=False):
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        # mini-batch queue
        self.mini_batch_queue = mini_batch_queue
        self.batch_size = batch_size
        self.output_image_size = output_image_size
        self.cropped_image_size = cropped_image_size
        self.diff_size = self.output_image_size - cropped_image_size
        self.img_macro = macro_data[0]['data']
        self.tgt_macro = macro_data[1]
        self.lbl_macro = macro_data[2]
        self.backend = backend
        self.dotransforms = dotransforms

        imsz = (cropped_image_size ** 2) * 3
        self.inputs = np.empty((imsz, self.batch_size), dtype='float32')
        self.num_processes = num_processes
        self.mean_img = mean_img
        self.predict = predict

    def jpeg_decoder(self, start_id, end_id, offset):
        # convert jpeg string to numpy array
        imdim = self.cropped_image_size
        meanimg = self.mean_img
        if self.predict:  # during prediction just use center crop & no flips
            csx = self.diff_size / 2
            csy = csx
            crop_mean_img = (meanimg[:, csx:csx + imdim, csy:csy + imdim])
        for i, jpeg_string in enumerate(self.img_macro[start_id:end_id]):
            img = Image.open(StringIO(jpeg_string))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            csx = np.random.randint(0, max(self.diff_size, 1))
            csy = np.random.randint(0, max(self.diff_size, 1))
            crop_mean_img = (meanimg[:, csx:csx + imdim, csy:csy + imdim])
            if not self.predict and self.dotransforms:  # for training
                # horizontal reflections of the image
                flip_horizontal = np.random.randint(0, 2)
                if flip_horizontal == 1:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

            img = img.crop((csx, csy, csx + imdim, csy + imdim))
            self.inputs[:, i + offset] = (
                np.transpose(np.array(
                    img, dtype='float32')[:, :, 0:3],
                    axes=[2, 0, 1])).reshape((-1))/255.

                # np.transpose(np.array(
                #     img, dtype='float32')[:, :, 0:3],
                #     axes=[2, 0, 1]) - crop_mean_img).reshape((-1))

    def run(self):
        # provide mini batch from macro batch
        s_idx = self.mb_id * self.batch_size
        e_idx = (self.mb_id + 1) * self.batch_size

        # single process decompression
        self.jpeg_decoder(s_idx, e_idx, 0)

        targets = None
        if self.tgt_macro is not None:
            targets = self.tgt_macro[:, s_idx:e_idx].copy()

        labels = {k: np.array(
                  self.lbl_macro[k][np.newaxis, s_idx:e_idx].copy(),
                  dtype=np.float32)
                  for k in self.lbl_macro.keys()}

        logger.debug('mini-batch decompress end %d', self.mb_id)

        self.mini_batch_queue.put([self.inputs, targets, labels])
        if not miniq.empty():
            di = miniq.get()
            di.start()
            miniq.task_done()
        else:
            global miniq_flag
            miniq_flag = False


class RingBuffer(object):

    '''
    ring buffer for GPU backend to assist in transferring data from host to gpu
    the buffers that live on host
    '''

    def __init__(self, max_size, batch_size, num_input_dims, num_tgt_dims,
                 label_list):
        self.max_size = max_size
        self.id = 0
        self.prev_id = 0
        tmp_input = np.empty((num_input_dims, batch_size), dtype='float32')

        tmp_targets = None
        if num_tgt_dims is not None:
            tmp_targets = np.empty((num_tgt_dims, batch_size), dtype='float32')

        tmp_labels = {}
        for lbl in label_list:
            tmp_labels[lbl] = np.empty((1, batch_size), dtype='float32')

        self.inputs_be = [GPUTensor(tmp_input) for i in range(max_size)]

        if tmp_targets is not None:
            self.targets_be = [GPUTensor(tmp_targets) for i in range(max_size)]

        self.labels_be = [{lbl: GPUTensor(tmp_labels[lbl])
                           for lbl in tmp_labels.keys()}
                          for i in range(max_size)]

        self.labels_be = []
        for i in range(max_size):
            self.inputs_be.append(GPUTensor(tmp_input))
            if tmp_targets is not None:
                self.targets_be.append(GPUTensor(tmp_targets))
            tmp_labels_be = {}
            for lbl in label_list:
                tmp_labels_be[lbl] = GPUTensor(tmp_labels[lbl])
            self.labels_be.append(tmp_labels_be)

    def add_item(self, inputs, targets, labels, backend):
        self.prev_id = self.id
        logger.debug('start add_item')

        # using ring buffer
        # TODO Change to use copy_from
        self.inputs_be[self.id].set_host_mat(inputs)
        self.inputs_be[self.id].copy_to_device()

        if targets is not None:
            self.targets_be[self.id].set_host_mat(targets)
            self.targets_be[self.id].copy_to_device()

        for lbl in labels.keys():
            self.labels_be[self.id][lbl].set_host_mat(labels[lbl])
            self.labels_be[self.id][lbl].copy_to_device()

        logger.debug('end add_item')
        self.id += 1
        if self.id == self.max_size:
            self.id = 0


class GPUTransfer(threading.Thread):

    '''
    thread to transfer data to GPU
    '''
    def __init__(self, mb_id, mini_batch_queue, gpu_queue, backend,
                 ring_buffer):
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        self.mini_batch_queue = mini_batch_queue
        self.gpu_queue = gpu_queue
        self.backend = backend
        self.ring_buffer = ring_buffer

    def run(self):
        logger.debug('backend mini-batch transfer start %d', self.mb_id)
        # threaded conversion of jpeg strings to numpy array
        # if no item in queue, wait
        inputs, targets, labels = self.mini_batch_queue.get(block=True)
        logger.debug("popped mini_batch_queue")

        if isinstance(self.backend, GPU):
            rbuf = self.ring_buffer
            rbuf.add_item(inputs, targets, labels, self.backend)
            self.gpu_queue.put([rbuf.inputs_be[rbuf.prev_id],
                                rbuf.targets_be[rbuf.prev_id],
                                rbuf.labels_be[rbuf.prev_id]])
        else:
            sbearray = self.backend.array
            inputs_be = sbearray(inputs)
            targets_be = None if targets is None else sbearray(targets)
            labels_be = sbearray(labels)
            self.gpu_queue.put([inputs_be, targets_be, labels_be])
        logger.debug('backend mini-batch transfer done %d', self.mb_id)
        self.mini_batch_queue.task_done()

        if not gpuq.empty():
            gt = gpuq.get()
            gt.start()
            try:
                gpuq.task_done()
            except:
                pass
        else:
            global gpuq_flag
            gpuq_flag = False


class Imageset(Dataset):

    """
    Sets up a macro batched imageset dataset.

    Assumes you have the data alreay partitioned and in macrobatch format

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk

    """
    def __init__(self, **kwargs):
        opt_param(self, ['start_train', 'end_train',
                         'start_val', 'end_val'], -1)
        opt_param(self, ['preprocess_done', 'dotransforms', 'dist_flag'],
                  False)
        opt_param(self, ['tdims'], 0)
        opt_param(self, ['label_list'], ['l_id'])
        self.__dict__.update(kwargs)
        req_param(self, ['save_dir'])
        req_param(self, ['label_list', 'cropped_image_size'])
        self.idims = (self.cropped_image_size ** 2) * 3

        if self.start_train != -1:
            # num train batches for this yaml file (<= total available)
            self.n_train_batches = self.end_train - self.start_train + 1
        if self.start_val != -1:
            # num validation batches for this yaml file (<= total available)
            self.n_val_batches = self.end_val - self.start_val + 1

    def load(self):
        pass

    def preprocess_images(self):
        # compute mean of all the images
        logger.info("preprocessing images (computing mean image)")
        self.mean_img = np.zeros((3, self.output_image_size,
                                  self.output_image_size), dtype='float32')
        t1 = time()
        for i in range(self.n_train_batches):
            logger.info("preprocessing macro-batch %d :", i)
            file_path = os.path.join(self.save_dir, 'data_batch_%d' % (i))
            jpeg_strings = my_unpickle(file_path)
            for jpeg_string in jpeg_strings['data']:
                img = Image.open(StringIO(jpeg_string))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                self.mean_img += np.transpose(np.array(
                    img, dtype='float32')[:, :, 0:3],
                    axes=[2, 0, 1])  # .reshape((-1, 1))
        logger.info("Time taken %f", time()-t1)
        self.mean_img = (self.mean_img /
                         (self.n_train_batches * self.output_batch_size))
        logger.info("done preprocessing images (computing mean image)")

    def get_next_macro_batch_id(self, macro_batch_index):
        next_macro_batch_id = macro_batch_index + 1
        if next_macro_batch_id > self.endb:
            next_macro_batch_id = self.startb
        return next_macro_batch_id

    def get_macro_batch(self):
        for i in range(self.num_iter_macro):
            self.macro_batch_onque = self.get_next_macro_batch_id(
                self.macro_batch_onque)
            self.file_name_queue.put(
                os.path.join(self.save_dir, 'data_batch_%d' % (
                    self.macro_batch_onque)))
            t = LoadFile(self.file_name_queue, self.macro_batch_queue)
            t.setDaemon(True)
            global macroq_flag
            if macroq_flag:
                macroq.put(t)
            else:
                t.start()
                # if macroq is not empty then set macroq_flag to True
                macroq_flag = True

        self.num_iter_macro = 1

    def get_next_mini_batch_id(self, mini_batch_index):
        next_mini_batch_id = mini_batch_index + 1
        if next_mini_batch_id >= self.num_minibatches_in_macro:
            next_mini_batch_id = 0
        return next_mini_batch_id

    def init_mini_batch_producer(self, batch_size, setname, predict=False):
        sn = 'val' if (setname == 'validation') else setname
        self.endb = getattr(self, 'end_' + sn)
        self.startb = getattr(self, 'start_' + sn)
        nrecs = self.output_batch_size * (self.endb - self.startb + 1)
        if self.startb == -1 or self.endb == -1:
            raise NotImplementedError("Must specify [start|end]"
                                      "_[train|val]_batch")
        num_batches = int(np.ceil((nrecs + 0.0) / batch_size))

        self.batch_size = batch_size
        self.batch_type = setname
        self.predict = predict

        if self.output_batch_size % batch_size != 0:
            raise ValueError('self.output_batch_size % batch_size != 0')
        else:
            self.num_minibatches_in_macro = self.output_batch_size / batch_size
        if isinstance(self.backend, GPU):
            self.ring_buffer = RingBuffer(max_size=self.ring_buffer_size,
                                          batch_size=batch_size,
                                          num_tgt_dims=self.tdims,
                                          num_input_dims=self.idims,
                                          label_list=self.label_list)
        self.file_name_queue = Queue.Queue()
        self.macro_batch_queue = Queue.Queue()
        self.mini_batch_queue = Queue.Queue()
        self.gpu_queue = Queue.Queue()
        global macroq, miniq, gpuq, macroq_flag, miniq_flag, gpuq_flag
        macroq = Queue.Queue()
        miniq = Queue.Queue()
        gpuq = Queue.Queue()
        macroq_flag = False
        miniq_flag = False
        gpuq_flag = False
        self.macro_batch_onque = self.endb
        self.mini_batch_onque = self.num_minibatches_in_macro - 1
        self.gpu_batch_onque = self.num_minibatches_in_macro - 1
        self.num_iter_mini = self.ring_buffer_size
        self.num_iter_gpu = self.ring_buffer_size
        self.num_iter_macro = self.ring_buffer_size

        if not self.preprocess_done:
            self.preprocess_images()
            self.preprocess_done = True

        return num_batches

    def del_queue(self, qname):
        while not qname.empty():
            qname.get()
            qname.task_done()

    def del_mini_batch_producer(self):
        # graceful ending of thread queues
        self.del_queue(self.file_name_queue)
        self.del_queue(self.macro_batch_queue)
        self.del_queue(self.mini_batch_queue)
        self.del_queue(self.gpu_queue)
        global macroq, miniq, gpuq
        self.del_queue(macroq)
        self.del_queue(miniq)
        self.del_queue(gpuq)
        del self.file_name_queue, self.macro_batch_queue
        del self.mini_batch_queue, self.gpu_queue

    def get_mini_batch(self, batch_idx):
        # threaded version of get_mini_batch
        # batch_idx is ignored
        gt_args = {'backend': self.backend, 'ring_buffer': self.ring_buffer,
                   'gpu_queue': self.gpu_queue}

        for i in range(self.num_iter_mini):
            self.mini_batch_onque = self.get_next_mini_batch_id(
                self.mini_batch_onque)
            if self.mini_batch_onque == 0:
                self.get_macro_batch()
                # deque next macro batch
                try:
                    self.macro_batch_queue.task_done()
                except:
                    # allow for the first get from macro_batch_queue
                    pass
                self.jpeg_strings = self.macro_batch_queue.get(block=True)
                self.targets_macro = None
                if 'target' in self.jpeg_strings:
                    self.targets_macro = self.jpeg_strings['target']
                self.labels_macro = {}
                for key in self.jpeg_strings['labels']:
                    self.labels_macro[key] = self.jpeg_strings['labels'][key]
            macro_data = [self.jpeg_strings, self.targets_macro,
                          self.labels_macro]
            di = DecompressImages(self.mini_batch_onque, self.mini_batch_queue,
                                  self.batch_size, self.output_image_size,
                                  self.cropped_image_size, macro_data,
                                  self.backend, self.num_processes,
                                  self.mean_img, self.predict)
            di.setDaemon(True)
            global miniq_flag
            if miniq_flag:
                miniq.put(di)
            else:
                di.start()
                miniq_flag = True

        for i in range(self.num_iter_gpu):
            self.gpu_batch_onque = self.get_next_mini_batch_id(
                self.gpu_batch_onque)
            gt_args['mb_id'] = self.gpu_batch_onque
            gt_args['mini_batch_queue'] = self.mini_batch_queue
            # gt = GPUTransfer(gt_args)
            gt = GPUTransfer(self.gpu_batch_onque,
                             self.mini_batch_queue, self.gpu_queue,
                             self.backend, self.ring_buffer)

            gt.setDaemon(True)
            global gpuq_flag
            if gpuq_flag:
                gpuq.put(gt)
            else:
                gt.start()
                gpuq_flag = True

        inputs_be, targets_be, labels_be = self.gpu_queue.get(block=True)
        self.gpu_queue.task_done()
        self.num_iter_mini = 1
        self.num_iter_gpu = 1

        return inputs_be, targets_be, labels_be

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False
