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
import tarfile
import cPickle
from PIL import Image
from StringIO import StringIO
import scipy.io
from random import shuffle
from time import time
from neon.datasets.dataset import Dataset
from neon.util.compat import MPI_INSTALLED
import sys
import threading
from collections import deque

logger = logging.getLogger(__name__)

def my_pickle(filename, data):
        with open(filename, "w") as fo:
            cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

class LoadFile(threading.Thread):
    def __init__(self, fname, file_queue):
        threading.Thread.__init__(self)
        self.fname = fname
        self.file_queue = file_queue
        
    def run(self):
        logger.info('%s: loading file', self.fname)
        self.file_queue.append(my_unpickle(self.fname))
        logger.info('%s: done loading file', self.fname)

class DecompressImages(threading.Thread):
    def __init__(self, mb_id, mb_queue, batch_size, output_image_size,
            jpeg_strings, targets_macro):
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        #mini-batch queue
        self.mb_queue = mb_queue
        self.batch_size = batch_size
        self.output_image_size = output_image_size
        self.jpeg_strings = jpeg_strings
        self.targets_macro = targets_macro
        
    def run(self):
        logger.info('mini-batch decompress start %d', self.mb_id)
        # provide mini batch from macro batch
        inputs = np.empty(
            ((self.output_image_size ** 2) * 3, self.batch_size),
            dtype='float32')
        start_idx = self.mb_id * self.batch_size
        end_idx = (self.mb_id + 1) * self.batch_size

        # convert jpeg string to numpy array
        for i, jpeg_string in enumerate(
                self.jpeg_strings['data'][start_idx:end_idx]):
            img = Image.open(StringIO(jpeg_string))
            if img.mode == 'L':  # greyscale
                logger.debug('greyscale image found... tiling')
                inputs[:, i, np.newaxis] = np.tile(
                    np.array(img, dtype='float32').reshape((-1, 1)), (3, 1))
            else:
                inputs[:, i, np.newaxis] = np.array(
                    img, dtype='float32').reshape((-1, 1))

        targets = self.targets_macro[:, start_idx:end_idx]

        self.mb_queue.append([inputs, targets])
        logger.info('mini-batch decompress end %d', self.mb_id)

class GPUTransfer(threading.Thread):
    def __init__(self, mb_id, mb_queue, gpu_queue, backend):
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        self.mb_queue = mb_queue
        self.gpu_queue = gpu_queue
        self.backend = backend

    def run(self):
        logger.info('backend mini-batch transfer start %d', self.mb_id)
        # threaded conversion of jpeg strings to numpy array
        if len(self.mb_queue) == 0:
            return
        inputs, targets = self.mb_queue.popleft()
        inputs_backend = self.backend.array(inputs)
        targets_backend = self.backend.array(targets)
        logger.info('backend mini-batch transfer done %d', self.mb_id)
        self.gpu_queue.append([inputs_backend, targets_backend])

class I1K(Dataset):

    """
    Sets up a ImageNet-1000 dataset.

    Attributes:
        url (str): where to find the source data
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk

    """
    url = "http://www.image-net.org/download-imageurls"

    def __init__(self, **kwargs):
        self.dist_flag = False
        self.macro_batched = False
        self.start_train_batch = -1
        self.end_train_batch = -1
        self.start_val_batch = -1
        self.end_val_batch = -1
        self.__dict__.update(kwargs)
        if self.dist_flag:
            raise NotImplementedError('Dist not implemented for I1K!')
            if MPI_INSTALLED:
                from mpi4py import MPI
                self.comm = MPI.COMM_WORLD
                # for now require that comm.size is a square and divides 32
                if self.comm.size not in [1, 4, 16]:
                    raise AttributeError('MPI.COMM_WORLD.size not compatible')
            else:
                raise AttributeError("dist_flag set but mpi4py not installed")
        if self.macro_batched:
            # macro batch queue
            self.macro_batch_queue = deque([])
            # mini batch queue
            self.mini_batch_queue = deque([])
            # gpu/backend queue
            self.gpu_queue = deque([])

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            # todo handle dist case
            # if self.dist_flag:
            #    self.adjust_for_dist()

            self.nclasses = 1000
            load_dir = os.path.join(self.load_path,
                                    self.__class__.__name__)
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            self.save_dir = save_dir
            # for now assuming that dataset is already there
            # ToS of imagenet prohibit distribution of URLs

            # based on krizhevsky's make-data.py
            ilsvrc_train_tar = os.path.join(
                load_dir, 'ILSVRC2012_img_train.tar')
            ilsvrc_validation_tar = os.path.join(
                load_dir, 'ILSVRC2012_img_val.tar')
            ilsvrc_devkit_tar = os.path.join(
                load_dir, 'ILSVRC2012_devkit_t12.tar.gz')
            for infile in (ilsvrc_train_tar, ilsvrc_validation_tar,
                           ilsvrc_devkit_tar):
                if not os.path.exists(infile):
                    raise IOError("%s not found.  Please ensure you have"
                                  "ImageNet downloaded.  More info here: %s",
                                  infile, self.url)
            labels_dic, label_names, validation_labels = self.parse_dev_meta(
                ilsvrc_devkit_tar)

            with self.open_tar(ilsvrc_train_tar, 'training tar') as tf:
                synsets = tf.getmembers()
                synset_tars = [
                    tarfile.open(fileobj=tf.extractfile(s)) for s in synsets]
                # subsampling the first n tar files for now
                # todo: delete this line
                synset_tars = synset_tars[:self.max_tar_file]
                logger.info("Loaded synset tars.")
                logger.info('Building training set image list '
                            '(this can take 10-20 minutes)...')

                train_jpeg_files = []
                for i, st in enumerate(synset_tars):
                    if i % 100 == 0:
                        logger.info("%d%% ...",
                                    int(round(100.0 * float(i) /
                                        len(synset_tars))))
                    train_jpeg_files += [st.extractfile(m)
                                         for m in st.getmembers()]
                    st.close()

                shuffle(train_jpeg_files)
                train_labels = [[labels_dic[jpeg.name[:9]]]
                                for jpeg in train_jpeg_files]
                logger.info("created list of jpg files")

                self.crop_to_square = True
                # todo: Number of threads to use for JPEG decompression and
                # image resizing.
                self.num_worker_threads = 8
                # macro batch size
                self.output_batch_size = 3072
                self.max_file_index = 3072
                jpeg_file_sample = train_jpeg_files[0:self.max_file_index]
                label_sample = train_labels[0:self.max_file_index]

                self.val_max_file_index = 3072
                val_label_sample = validation_labels[0:self.val_max_file_index]

                # todo 2: implement macro batching [will require changing model
                # code]
                if self.macro_batched:
                    # Write training batches
                    self.num_train_macro_batches = self.write_batches(
                        os.path.join(save_dir, 'macro_batches_'
                                     + str(self.output_image_size)),
                        'training', 0,
                        label_sample, jpeg_file_sample)
                    with self.open_tar(ilsvrc_validation_tar,
                                       'validation tar') as tf:
                        validation_jpeg_files = sorted(
                            [tf.extractfile(m) for m in tf.getmembers()],
                            key=lambda x: x.name)
                        val_file_sample = validation_jpeg_files[
                            0:self.val_max_file_index]
                        self.num_val_macro_batches = self.write_batches(
                            os.path.join(save_dir, 'macro_batches_'
                                         + str(self.output_image_size)),
                            'validation', 0,
                            val_label_sample,
                            val_file_sample)
                    self.cur_train_macro_batch = 0
                    self.cur_train_mini_batch = 0
                    self.cur_val_macro_batch = 0
                    self.cur_val_mini_batch = 0
                else:
                    # one big batch (no macro-batches)
                    # todo 1: resize in a multithreaded manner
                    jpeg_mat = self.resize_jpeg(
                        [jpeg.read() for jpeg in jpeg_file_sample],
                        as_string=False)

                    self.inputs['train'] = jpeg_mat
                    # convert labels to one hot
                    tmp = np.zeros(
                        (self.nclasses, self.max_file_index), dtype='float32')
                    for col in range(self.nclasses):
                        tmp[col] = label_sample == col
                    self.targets['train'] = tmp
                    logger.info('done loading training data')

                    with self.open_tar(ilsvrc_validation_tar,
                                       'validation tar') as tf:
                        validation_jpeg_files = sorted(
                            [tf.extractfile(m) for m in tf.getmembers()],
                            key=lambda x: x.name)
                        val_file_sample = validation_jpeg_files[
                            0:self.val_max_file_index]
                        jpeg_mat = self.resize_jpeg(
                            [jpeg.read() for jpeg in val_file_sample],
                            as_string=False)

                        self.inputs['test'] = jpeg_mat
                        tmp = np.zeros(
                            (self.nclasses, self.max_file_index),
                            dtype='float32')
                        for col in range(self.nclasses):
                            tmp[col] = val_label_sample == col
                        self.targets['test'] = tmp

                    logger.info("done loading imagenet data")
                    self.format()
        else:
            raise AttributeError('repo_path not specified in config')

    def resize_jpeg(self, jpeg_strings, as_string=False):
        if as_string:
            tgt = []
        else:
            # as numpy array, row order
            tgt = np.empty(
                (len(jpeg_strings), (self.output_image_size ** 2) * 3),
                dtype='float32')
        for i, jpeg_string in enumerate(jpeg_strings):
            img = Image.open(StringIO(jpeg_string))

            # resize
            min_dim = np.min(img.size)
            scale_factor = np.float(self.output_image_size) / min_dim
            new_w = np.int(np.rint(scale_factor * img.size[0]))
            new_h = np.int(np.rint(scale_factor * img.size[1]))
            img = img.resize((new_w, new_h))  # todo: interp mode?

            # crop
            if self.crop_to_square:
                crop_start_x = (new_w - self.output_image_size) / 2
                crop_start_y = (new_h - self.output_image_size) / 2
                img = img.crop((crop_start_x, crop_start_y,
                                crop_start_x + self.output_image_size,
                                crop_start_y + self.output_image_size))
            else:
                raise NotImplementedError
            if as_string:
                f = StringIO()
                img.save(f, "JPEG")
                tgt.append(f.getvalue())
            else:
                # this is still in row order
                if img.mode == 'L':  # greyscale
                    logger.debug('greyscale image found... tiling')
                    tgt[i] = np.tile(
                        np.array(img, dtype='float32').reshape((1, -1)), 3)
                else:
                    tgt[i] = np.array(img, dtype='float32').reshape((1, -1))

        return tgt

    def get_next_macro_batch_id(self, batch_type, macro_batch_index):
        next_macro_batch_id = macro_batch_index + 1
        if batch_type == 'training':
            if (next_macro_batch_id >= self.num_train_macro_batches
                    and self.end_train_batch == -1):
                next_macro_batch_id = 0
            elif next_macro_batch_id > self.end_train_batch:
                next_macro_batch_id = self.start_train_batch
        elif batch_type == 'validation':
            if (next_macro_batch_id >= self.num_val_macro_batches and
                    self.end_val_batch == -1):
                next_macro_batch_id = 0
            elif next_macro_batch_id > self.end_val_batch:
                next_macro_batch_id = self.start_val_batch
        return next_macro_batch_id

    def get_macro_batch(self, batch_type, macro_batch_index,
                        raw_targets=False):
        batch_path = os.path.join(
            self.save_dir, 'macro_batches_' + str(self.output_image_size),
            '%s_batch_%d' % (batch_type, macro_batch_index))
        j = 0
        #if macro_batch_queue is empty... 
        if len(self.macro_batch_queue) == 0:
            #load 2 + 1 macro batches
            thread1 = LoadFile(os.path.join(batch_path, '%s_batch_%d.%d' % (
                batch_type, macro_batch_index, j / self.output_batch_size)),
                self.macro_batch_queue)
            thread1.start()
            self.macro_batch_onque = self.get_next_macro_batch_id(batch_type,
                macro_batch_index)
            thread2 = LoadFile(os.path.join(batch_path, '%s_batch_%d.%d' % (
                batch_type,
                self.macro_batch_onque,
                j / self.output_batch_size)),
                self.macro_batch_queue)
            thread2.start() #load the second macro batch asynchronously
            self.macro_batch_onque = self.get_next_macro_batch_id(batch_type,
                self.macro_batch_onque)
            thread3 = LoadFile(os.path.join(batch_path, '%s_batch_%d.%d' % (
                batch_type,
                self.macro_batch_onque,
                j / self.output_batch_size)),
                self.macro_batch_queue)
            thread3.start() # this is asynchronous
            thread1.join() # first two macro batches will be sequential
            thread2.join()
        else:
            self.macro_batch_onque = self.get_next_macro_batch_id(batch_type,
                self.macro_batch_onque)
            thread1 = LoadFile(os.path.join(batch_path, '%s_batch_%d.%d' % (
                batch_type,
                self.macro_batch_onque,
                j / self.output_batch_size)),
                self.macro_batch_queue)
            thread1.start()

        #pop a macro batch off the thread queue
        self.jpeg_strings = self.macro_batch_queue.popleft()
        
        # todo: this part could also be threaded for speedup
        # during run time extract labels
        labels = self.jpeg_strings['labels']

        if not raw_targets:
            self.targets_macro = np.zeros(
                (self.nclasses, self.output_batch_size), dtype='float32')
            for col in range(self.nclasses):
                self.targets_macro[col] = labels == col
        else:
            self.targets_macro = np.asarray(labels).reshape((1, -1))

    def get_next_mini_batch_id(self, batch_type, mini_batch_index):
        next_mini_batch_id = mini_batch_index + 1
        if batch_type == 'training':
            if next_mini_batch_id >= self.num_minibatches_in_macro:
                next_mini_batch_id = 0
        elif batch_type == 'validation':
            if next_mini_batch_id >= self.num_minibatches_in_macro:
                next_mini_batch_id = 0
        return next_mini_batch_id

    def get_mini_batch(self, batch_size, batch_type, raw_targets=False):
        if self.output_batch_size % batch_size != 0:
            raise ValueError('self.output_batch_size % batch_size != 0')
        else:
            self.num_minibatches_in_macro = self.output_batch_size / batch_size

        # keep track of most recent batch and return the next batch
        if batch_type == 'training':
            cur_mini_batch_id = self.cur_train_mini_batch
            if cur_mini_batch_id == 0:
                # when cur_mini_batch is 0 load macro batch
                logger.info("train processing macro batch: %d",
                            self.cur_train_macro_batch)
                self.get_macro_batch(
                    batch_type, self.cur_train_macro_batch, raw_targets)
                self.cur_train_macro_batch = self.get_next_macro_batch_id(
                    'training', self.cur_train_macro_batch)
        elif batch_type == 'validation':
            cur_mini_batch_id = self.cur_val_mini_batch
            if cur_mini_batch_id == 0:
                logger.info(
                    "val processing macro batch: %d", self.cur_val_macro_batch)
                self.get_macro_batch(
                    batch_type, self.cur_val_macro_batch, raw_targets)
                self.cur_val_macro_batch = self.get_next_macro_batch_id(
                    'validation', self.cur_val_macro_batch)
        else:
            raise ValueError('Invalid batch_type in get_batch')

        if len(self.mini_batch_queue) == 0:
            thread1 = DecompressImages(cur_mini_batch_id,
                self.mini_batch_queue, batch_size, self.output_image_size,
                self.jpeg_strings, self.targets_macro)
            thread1.start()
            self.mini_batch_onque = self.get_next_mini_batch_id(batch_type,
                cur_mini_batch_id)
            thread2 = DecompressImages(self.mini_batch_onque,
                self.mini_batch_queue, batch_size, self.output_image_size,
                self.jpeg_strings, self.targets_macro)
            thread2.start()
            self.mini_batch_onque = self.get_next_mini_batch_id(batch_type,
                self.mini_batch_onque)
            thread3 = DecompressImages(self.mini_batch_onque,
                self.mini_batch_queue, batch_size, self.output_image_size,
                self.jpeg_strings, self.targets_macro)
            thread3.start()
            thread1.join()
            thread2.join()
        else:
            tmp = self.mini_batch_onque
            self.mini_batch_onque = self.get_next_mini_batch_id(batch_type,
                self.mini_batch_onque)
            # todo: no wrap around to next macro batch for mini currently
            if self.mini_batch_onque != 0:
                thread1 = DecompressImages(self.mini_batch_onque,
                    self.mini_batch_queue, batch_size, self.output_image_size,
                    self.jpeg_strings, self.targets_macro)
                thread1.start()
            else:
                self.mini_batch_onque = tmp

        
        # serialize
        # todo:
        # test file reading speeds 3M, 30M, 90M, 300M
        # multithreaded reads
        # multiple MPI processes reading diff files from same disk

        # TODO: resize 256x256 image to 224x224 here

        # todo: run this with page locked memory
        # thread host -> gpu transfers onto a queue
        if len(self.gpu_queue) == 0:
            thread1 = GPUTransfer(self.cur_train_mini_batch,
                self.mini_batch_queue, self.gpu_queue, self.backend)
            self.gpu_batch_onque = self.get_next_mini_batch_id(batch_type,
                self.cur_train_mini_batch)
            thread2 = GPUTransfer(self.gpu_batch_onque,
                self.mini_batch_queue, self.gpu_queue, self.backend)
            thread1.start()
            thread2.start()
            thread1.join()
        else:
            tmp = self.gpu_batch_onque
            self.gpu_batch_onque = self.get_next_mini_batch_id(batch_type,
                self.gpu_batch_onque)
            if self.gpu_batch_onque != 0:
                thread1 = GPUTransfer(self.gpu_batch_onque,
                    self.mini_batch_queue, self.gpu_queue, self.backend)
                thread1.start()
            else:
                self.gpu_batch_onque = tmp
        
        if batch_type == 'training':
            self.cur_train_mini_batch = self.get_next_mini_batch_id(batch_type, cur_mini_batch_id)
        elif batch_type == 'validation':
            self.cur_val_mini_batch = self.get_next_mini_batch_id(batch_type, cur_mini_batch_id)

        inputs_backend, targets_backend = self.gpu_queue.popleft()
        return inputs_backend, targets_backend

    # code below from Alex Krizhevsky's cuda-convnet2 library, make-data.py
    # Copyright 2014 Google Inc. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #    http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    ##########################################################################
    def open_tar(self, path, name):
        if not os.path.exists(path):
            logger.error("ILSVRC 2012 %s not found at %s.",
                         "Make sure to set ILSVRC_SRC_DIR correctly at the",
                         "top of this file (%s)." % (name, path, sys.argv[0]))
            sys.exit(1)
        return tarfile.open(path)

    def makedir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def parse_dev_meta(self, ilsvrc_devkit_tar):
        tf = self.open_tar(ilsvrc_devkit_tar, 'devkit tar')
        fmeta = tf.extractfile(
            tf.getmember('ILSVRC2012_devkit_t12/data/meta.mat'))
        meta_mat = scipy.io.loadmat(StringIO(fmeta.read()))
        labels_dic = dict((m[0][1][0], m[0][0][0][
                          0] - 1) for m in meta_mat['synsets']
                          if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
        label_names_dic = dict((m[0][1][0], m[0][2][0]) for m in meta_mat[
                               'synsets'] if m[0][0][0][0] >= 1
                               and m[0][0][0][0] <= 1000)
        label_names = [tup[1] for tup in sorted(
            [(v, label_names_dic[k]) for k, v in labels_dic.items()],
            key=lambda x:x[0])]

        fval_ground_truth = tf.extractfile(tf.getmember(
            'ILSVRC2012_devkit_t12/data/' +
            'ILSVRC2012_validation_ground_truth.txt'))
        validation_ground_truth = [
            [int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
        tf.close()
        return labels_dic, label_names, validation_ground_truth

    # following functions are for creating macrobatches
    def partition_list(self, l, partition_size):
        divup = lambda a, b: (a + b - 1) / b
        return [l[i * partition_size:(i + 1) * partition_size]
                for i in xrange(divup(len(l), partition_size))]

    def write_batches(self, target_dir, name, start_batch_num, labels,
                      jpeg_files):
        jpeg_files = self.partition_list(jpeg_files, self.output_batch_size)
        labels = self.partition_list(labels, self.output_batch_size)
        self.makedir(target_dir)
        logger.debug("Writing %s batches..." % name)
        for i, (labels_batch, jpeg_file_batch) in enumerate(
                zip(labels, jpeg_files)):
            t = time()
            jpeg_strings = self.resize_jpeg(
                [jpeg.read() for jpeg in jpeg_file_batch], as_string=True)
            batch_path = os.path.join(
                target_dir, '%s_batch_%d' % (name, start_batch_num + i))
            self.makedir(batch_path)
            # no subbatch support for now; do we really need them?
            # for j in xrange(0, len(labels_batch),
            # self.OUTPUT_SUB_BATCH_SIZE):
            j = 0
            my_pickle(os.path.join(batch_path, '%s_batch_%d.%d' %
                        (name, start_batch_num + i, j /
                            self.output_batch_size)),
                        {'data': jpeg_strings[j:j + self.output_batch_size],
                         'labels': labels_batch[j:j + self.output_batch_size]})
            logger.debug("Wrote %s (%s batch %d of %d) (%.2f sec)" %
                         (batch_path, name, i + 1,
                          len(jpeg_files), time() - t))
        return i + 1
