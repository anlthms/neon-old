# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
ImageNet 1k dataset
More information at: http://www.image-net.org/download-imageurls
Sign up for an ImageNet account to download the dataset!
"""

import logging
import multiprocessing as mp
import os
from random import shuffle
import sys
import tarfile
import threading
from time import time

import numpy as np

from neon.datasets.dataset import Dataset
from neon.util.compat import range, pickle, queue, StringIO

logger = logging.getLogger(__name__)

# prefix for directory name where macro_batches are stored
prefix_macro = 'macro_batches_'
# global queues to start threads
macroq = queue.Queue()
miniq = queue.Queue()
gpuq = queue.Queue()
macroq_flag = False
miniq_flag = False
gpuq_flag = False

BDTYPE = 'float16'
logger.warning("i1k with hardcoded dtype %s" ,BDTYPE)

def my_pickle(filename, data):
    with open(filename, "w") as fo:
        pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)


def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = pickle.load(fo)
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
                 cropped_image_size, jpeg_strings, targets_macro, backend,
                 num_processes, mean_img, predict):
        from PIL import Image
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        self.image = Image
        # mini-batch queue
        self.mini_batch_queue = mini_batch_queue
        self.batch_size = batch_size
        self.output_image_size = output_image_size
        self.cropped_image_size = cropped_image_size
        self.diff_size = self.output_image_size - self.cropped_image_size
        self.jpeg_strings = jpeg_strings
        self.targets_macro = targets_macro
        self.backend = backend
        # if using multiprocessing [not working yet]
        # self.inputs = mp.RawArray(ctypes.c_float,
        #     ((self.cropped_image_size ** 2) * 3 * self.batch_size))
        self.inputs = np.empty(
            ((self.cropped_image_size ** 2) * 3, self.batch_size),
            dtype=BDTYPE)
        self.num_processes = num_processes
        self.mean_img = mean_img
        self.predict = predict

    def jpeg_decoder(self, start_id, end_id, offset):
        # convert jpeg string to numpy array
        if self.predict:  # during prediction just use center crop & no flips
            csx = self.diff_size / 2
            csy = csx
            crop_mean_img = (
                self.mean_img[:, csx:csx + self.cropped_image_size,
                              csy:csy + self.cropped_image_size])
        for i, jpeg_string in enumerate(
                self.jpeg_strings['data'][start_id:end_id]):
            img = self.image.open(StringIO(jpeg_string))
            if not self.predict:  # for training
                # translations of image
                csx = np.random.randint(0, self.diff_size)
                csy = np.random.randint(0, self.diff_size)
                # horizontal reflections of the image
                flip_horizontal = np.random.randint(0, 2)
                if flip_horizontal == 1:
                    img = img.transpose(self.image.FLIP_LEFT_RIGHT)
                crop_mean_img = (
                    self.mean_img[:, csx:csx + self.cropped_image_size,
                                  csy:csy + self.cropped_image_size])
            img = img.crop((csx, csy,
                            csx + self.cropped_image_size,
                            csy + self.cropped_image_size))
            if(img.mode != 'RGB'):
                img = img.convert('RGB')
            self.inputs[:, i + offset] = (
                np.transpose(np.array(
                    img, dtype=BDTYPE)[:, :, 0:3],
                    axes=[2, 0, 1]) - crop_mean_img).reshape((-1))  / 128. - 1.  # urs wants this to be normalized # or not!

    def run(self):
        # provide mini batch from macro batch
        start_idx = self.mb_id * self.batch_size
        end_idx = (self.mb_id + 1) * self.batch_size

        # if using multiprocessing [not working yet]
        # start_list = range(
        #     start_idx, end_idx, self.batch_size / self.num_processes)
        # end_list = range(start_idx + self.batch_size / self.num_processes,
        #                  end_idx + 1, self.batch_size / self.num_processes)
        # offset_list = range(
        #     0, self.batch_size, self.batch_size / self.num_processes)
        # procs = [mp.Process(target=self.jpeg_decoder, args=[x1, x2, x3,
        #          self.inputs]) for (
        #     x1, x2, x3) in zip(start_list, end_list, offset_list)]
        # for proc in procs:
        #     proc.daemon = True
        #     proc.start()

        # single process decompression
        self.jpeg_decoder(start_idx, end_idx, 0)

        targets = self.targets_macro[:, start_idx:end_idx].copy()
        # [proc.join() for proc in procs]

        logger.debug('mini-batch decompress end %d', self.mb_id)

        self.mini_batch_queue.put([self.inputs, targets])
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

    def __init__(self, max_size, batch_size, num_targets, num_input_dims,
                 backend):
        self.max_size = max_size
        self.id = 0
        self.prev_id = 0
        tmp_input = np.empty((num_input_dims, batch_size), dtype=BDTYPE)
        tmp_target = np.empty((num_targets, batch_size), dtype=BDTYPE)

        self.inputs_backend = []
        self.targets_backend = []
        for i in range(max_size):
            self.inputs_backend.append(backend.array(tmp_input))
            self.targets_backend.append(backend.array(tmp_target))

    def add_item(self, inputs, targets, backend):
        self.prev_id = self.id
        logger.debug('start add_item')

        # using ring buffer
        backend.copy_from(self.inputs_backend[self.id], inputs)
        backend.copy_from(self.targets_backend[self.id], targets)

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
        from neon.backends.cpu import CPU
        logger.debug('backend mini-batch transfer start %d', self.mb_id)
        # threaded conversion of jpeg strings to numpy array
        # if no item in queue, wait
        inputs, targets = self.mini_batch_queue.get(block=True)
        logger.debug("popped mini_batch_queue")

        if not isinstance(self.backend, CPU):
            self.ring_buffer.add_item(inputs, targets, self.backend)
            self.gpu_queue.put([self.ring_buffer.inputs_backend[
                                self.ring_buffer.prev_id],
                                self.ring_buffer.targets_backend[
                                    self.ring_buffer.prev_id]])
        else:
            inputs_backend = self.backend.array(inputs)
            targets_backend = self.backend.array(targets)
            self.gpu_queue.put([inputs_backend, targets_backend])
        logger.debug('backend mini-batch transfer done %d', self.mb_id)
        self.mini_batch_queue.task_done()

        if not gpuq.empty():
            gt = gpuq.get()
            gt.start()
            gpuq.task_done()
        else:
            global gpuq_flag
            gpuq_flag = False


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
        from PIL import Image
        self.image = Image
        self.dist_flag = False
        self.start_train_batch = -1
        self.end_train_batch = -1
        self.start_val_batch = -1
        self.end_val_batch = -1
        self.preprocess_done = False
        self.__dict__.update(kwargs)
        self.repo_path = os.path.expandvars(os.path.expanduser(self.repo_path))

        if not hasattr(self, 'save_dir'):
            self.save_dir = os.path.join(self.repo_path,
                                         self.__class__.__name__)

        if self.start_train_batch != -1:
            # number of batches to train for this yaml file (<= total
            # available)
            self.n_train_batches = self.end_train_batch - \
                self.start_train_batch + 1
        if self.start_val_batch != -1:
            # number of batches to validate for this yaml file (<= total
            # available)
            self.n_val_batches = self.end_val_batch - \
                self.start_val_batch + 1

    def initialize(self):
        # perform additional setup that can't be done at initial construction
        if self.dist_flag:
            self.comm = self.backend.comm
            if self.comm.size not in [1, 4, 16]:
                raise AttributeError('MPI.COMM_WORLD.size not compatible')

    def load(self):
        if 'repo_path' in self.__dict__:
            # todo handle dist case
            # if self.dist_flag:
            #    self.adjust_for_dist()

            self.load_path = os.path.expandvars(os.path.expanduser(
                self.load_path))
            load_dir = os.path.join(self.load_path,
                                    self.__class__.__name__)
            self.save_dir = os.path.expandvars(os.path.expanduser(
                self.save_dir))
            save_dir = self.save_dir
            if os.path.exists(os.path.join(save_dir, prefix_macro + str(
                    self.output_image_size))):
                # delete load_dir if want to reload/reprocess dataset
                return

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
            # pickle the labels dic
            self.labels_dic = labels_dic

            with self.open_tar(ilsvrc_train_tar, 'training tar') as tf:
                synsets = tf.getmembers()
                synset_tars = [
                    tarfile.open(fileobj=tf.extractfile(s)) for s in synsets]
                # subsampling the first n tar files for now
                self.nclasses = self.max_tar_file
                synset_tars = synset_tars[:self.max_tar_file]

                logger.info("Loaded synset tars.")
                logger.info('Building training set image list '
                            '(this can take 10-45 minutes)...')

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
                # macro batch size
                logger.info("total number of training files = %d",
                            len(train_jpeg_files))
                self.max_file_index = (self.output_batch_size *
                                       self.num_train_macro_batches)

                jpeg_file_sample = train_jpeg_files[0:self.max_file_index]
                label_sample = train_labels[0:self.max_file_index]
                self.val_max_file_index = (self.output_batch_size *
                                           self.num_val_macro_batches)

                # this may not be most efficient
                flat_labels = [
                    item for sublist in train_labels for item in sublist]
                flat_labels = np.unique(flat_labels)
                print(flat_labels)
                for i in range(self.max_file_index):
                    for j in range(self.max_tar_file):
                        if label_sample[i][0] == flat_labels[j]:
                            label_sample[i] = [j]
                val_file_indices = []
                val_label_sample = []
                for i in range(len(validation_labels)):
                    for j in range(self.max_tar_file):
                        if validation_labels[i][0] == flat_labels[j]:
                            val_file_indices.append(i)
                            val_label_sample.append([j])

                # Write training batches
                self.num_train_macro_batches = self.write_batches(
                    os.path.join(save_dir, prefix_macro
                                 + str(self.output_image_size)),
                    'training', 0,
                    label_sample, jpeg_file_sample)
                with self.open_tar(ilsvrc_validation_tar,
                                   'validation tar') as tf:
                    validation_jpeg_files = sorted(
                        [tf.extractfile(m) for m in tf.getmembers()],
                        key=lambda x: x.name)
                    val_file_sample = [validation_jpeg_files[i]
                                       for i in val_file_indices]
                    # shuffle the validation file indices and corresponding
                    # labels before subsampling?
                    tmp_zip = zip(val_file_sample, val_label_sample)
                    shuffle(tmp_zip)
                    val_file_sample = [e[0] for e in tmp_zip]
                    val_file_sample = val_file_sample[
                        0:self.val_max_file_index]
                    val_label_sample = [e[1] for e in tmp_zip]
                    val_label_sample = val_label_sample[
                        0:self.val_max_file_index]
                    self.num_val_macro_batches = self.write_batches(
                        os.path.join(save_dir, prefix_macro
                                     + str(self.output_image_size)),
                        'validation', 0,
                        val_label_sample,
                        val_file_sample)
        else:
            raise AttributeError('repo_path not specified in config')

    def resize_jpeg(self, jpeg_strings, as_string=False):
        # currently just does crop and resize
        # reflections and translations happen on the fly in jpeg_decoder(...)
        if as_string:
            tgt = []
        else:
            # as numpy array, row order
            tgt = np.empty(
                (len(jpeg_strings), (self.output_image_size ** 2) * 3),
                dtype=BDTYPE)
        for i, jpeg_string in enumerate(jpeg_strings):
            img = self.image.open(StringIO(jpeg_string))

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
                raise NotImplementedError('returning as np.array not '
                                          'supported')
            # this is still in row order
            # if img.mode == 'L':  # greyscale
            #         logger.debug('greyscale image found... tiling')
            #         tgt[i] = np.tile(
            #             np.array(img, dtype=BDTYPE).reshape((1, -1)), 3)
            #     else:
            #         tgt[i] = np.array(img, dtype=BDTYPE).reshape((1, -1))

        return tgt

    def preprocess_images(self):
        # compute mean of all the images
        logger.info("preprocessing images (computing mean image)")
        self.mean_img = np.zeros((3, self.output_image_size,
                                  self.output_image_size), dtype=BDTYPE)
        return
        for i in range(self.n_train_batches):
            logger.info("preprocessing macro-batch %d :", i)
            batch_path = os.path.join(self.save_dir,
                                      prefix_macro +
                                      str(self.output_image_size),
                                      '%s_batch_%d' % ('training', i))
            j = 0
            file_path = os.path.join(batch_path, '%s_batch_%d.%d' % (
                'training', i, j / self.output_batch_size))
            jpeg_strings = my_unpickle(file_path)
            for jpeg_string in jpeg_strings['data']:
                img = self.image.open(StringIO(jpeg_string))
                if(img.mode != 'RGB'):
                    img = img.convert('RGB')
                self.mean_img += np.transpose(np.array(
                    img, dtype=BDTYPE)[:, :, 0:3],
                    axes=[2, 0, 1])  # .reshape((-1, 1))

        self.mean_img = (self.mean_img /
                         (self.n_train_batches * self.output_batch_size))
        logger.info("done preprocessing images (computing mean image)")

    def get_next_macro_batch_id(self, macro_batch_index):
        next_macro_batch_id = macro_batch_index + 1
        if next_macro_batch_id > self.endb:
            next_macro_batch_id = self.startb
        return next_macro_batch_id

    def get_macro_batch(self):
        j = 0
        for i in range(self.num_iter_macro):
            self.macro_batch_onque = self.get_next_macro_batch_id(
                self.macro_batch_onque)
            batch_path = os.path.join(
                self.save_dir, prefix_macro + str(self.output_image_size),
                '%s_batch_%d' % (self.batch_type, self.macro_batch_onque))
            self.file_name_queue.put(
                os.path.join(batch_path, '%s_batch_%d.%d' % (
                    self.batch_type, self.macro_batch_onque,
                    j / self.output_batch_size)))
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
        from neon.backends.cpu import CPU
        sn = 'val' if (setname == 'validation') else setname
        self.endb = getattr(self, 'end_' + sn + '_batch')
        self.startb = getattr(self, 'start_' + sn + '_batch')
        nrecs = self.output_batch_size * (self.endb - self.startb + 1)
        if self.startb == -1 or self.endb == -1:
            raise NotImplementedError("Must specify [start|end]"
                                      "_[train|val]_batch")
        num_batches = int(np.ceil((nrecs + 0.0) / batch_size))

        self.batch_size = batch_size
        self.batch_type = 'training' if (setname == 'train') else setname
        self.predict = predict
        self.nclasses = self.max_tar_file

        if self.output_batch_size % batch_size != 0:
            raise ValueError('self.output_batch_size % batch_size != 0')
        else:
            self.num_minibatches_in_macro = self.output_batch_size / batch_size
        if not isinstance(self.backend, CPU):
            self.ring_buffer = RingBuffer(max_size=self.ring_buffer_size,
                                          batch_size=batch_size,
                                          num_targets=self.nclasses,
                                          num_input_dims=(
                                              self.cropped_image_size ** 2)
                                          * 3,
                                          backend=self.backend)
        self.file_name_queue = queue.Queue()
        self.macro_batch_queue = queue.Queue()
        self.mini_batch_queue = queue.Queue()
        self.gpu_queue = queue.Queue()
        global macroq, miniq, gpuq, macroq_flag, miniq_flag, gpuq_flag
        macroq = queue.Queue()
        miniq = queue.Queue()
        gpuq = queue.Queue()
        macroq_flag = False
        miniq_flag = False
        gpuq_flag = False
        self.macro_batch_onque = self.endb
        # copies for get_mini_batch2()
        self.macro_batch_onque2 = self.endb
        self.mini_batch_onque2 = self.num_minibatches_in_macro - 1

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

    # def get_mini_batch2(self, batch_idx):
    #     # from neon.backends.gpu import GPUTensor
    #     # non threaded version of get_mini_batch for debugging
    #     self.mini_batch_onque2 = self.get_next_mini_batch_id(
    #         self.mini_batch_onque2)
    #     j = 0

    #     if self.mini_batch_onque2 == 0:
    #         self.macro_batch_onque2 = self.get_next_macro_batch_id(
    #             self.macro_batch_onque2)
    #         batch_path = os.path.join(
    #             self.save_dir, prefix_macro + str(self.output_image_size),
    #             '%s_batch_%d' % (self.batch_type, self.macro_batch_onque2))
    #         fname = os.path.join(batch_path, '%s_batch_%d.%d' % (
    #             self.batch_type, self.macro_batch_onque2,
    #             j / self.output_batch_size))
    #         self.jpeg_strings2 = my_unpickle(fname)

    #         labels = self.jpeg_strings2['labels']
    #         # flatten the labels list of lists to a single list
    #         labels = [item for sublist in labels for item in sublist]
    #         labels = np.asarray(labels, dtype=BDTYPE)  # -1.
    #         # if not self.raw_targets:
    #         self.targets_macro2 = np.zeros(
    #             (self.nclasses, self.output_batch_size), dtype=BDTYPE)
    #         for col in range(self.nclasses):
    #             self.targets_macro2[col] = labels == col
    #         # else:
    #         #     self.targets_macro2 = labels.reshape((1, -1))

    #     # provide mini batch from macro batch
    #     start_idx = self.mini_batch_onque2 * self.batch_size
    #     end_idx = (self.mini_batch_onque2 + 1) * self.batch_size

    #     targets = self.targets_macro2[:, start_idx:end_idx].copy()
    #     self.jpeg_decoder(start_idx, end_idx)

    #     return GPUTensor(self.inputs), GPUTensor(targets)

    def get_mini_batch(self, batch_idx):
        # threaded version of get_mini_batch
        # batch_idx is ignored
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
                labels = self.jpeg_strings['labels']
                # flatten the labels list of lists to a single list
                labels = [item for sublist in labels for item in sublist]
                labels = np.asarray(labels, dtype=BDTYPE)
                # if not self.raw_targets:
                self.targets_macro = np.zeros((self.nclasses,
                                               self.output_batch_size),
                                              dtype=BDTYPE)
                for col in range(self.nclasses):
                    self.targets_macro[col] = labels == col
                # else:
                #     self.targets_macro = labels.reshape((1, -1))

            di = DecompressImages(self.mini_batch_onque, self.mini_batch_queue,
                                  self.batch_size, self.output_image_size,
                                  self.cropped_image_size, self.jpeg_strings,
                                  self.targets_macro, self.backend,
                                  self.num_processes, self.mean_img,
                                  self.predict)
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

        inputs_backend, targets_backend = self.gpu_queue.get(block=True)
        self.gpu_queue.task_done()
        self.num_iter_mini = 1
        self.num_iter_gpu = 1

        return inputs_backend, targets_backend

    def jpeg_decoder(self, start_id, end_id):
        # helps with non-threaded decoding of jpegs
        # convert jpeg string to numpy array
        self.inputs = np.empty(
            ((self.cropped_image_size ** 2) * 3, self.batch_size),
            dtype=BDTYPE)
        self.diff_size = self.output_image_size - self.cropped_image_size
        if not self.predict:
            # use predict for training vs testing performance
            for i, jpeg_string in enumerate(
                    self.jpeg_strings2['data'][start_id:end_id]):
                img = self.image.open(StringIO(jpeg_string))
                # translations of image
                csx = np.random.randint(0, self.diff_size)
                csy = np.random.randint(0, self.diff_size)
                img = img.crop((csx, csy,
                                csx + self.cropped_image_size,
                                csy + self.cropped_image_size))
                # horizontal reflections of the image
                flip_horizontal = np.random.randint(0, 2)
                if flip_horizontal == 1:
                    img = img.transpose(self.image.FLIP_LEFT_RIGHT)
                if(img.mode != 'RGB'):
                    img = img.convert('RGB')
                crop_mean_img = (
                    self.mean_img[:, csx:csx + self.cropped_image_size,
                                  csy:csy + self.cropped_image_size])
                self.inputs[:, i, np.newaxis] = (
                    np.transpose(np.array(
                        img, dtype=BDTYPE)[:, :, 0:3],
                        axes=[2, 0, 1]) - crop_mean_img).reshape((-1, 1))
        else:
            csx = self.diff_size / 2
            csy = csx
            crop_mean_img = (
                self.mean_img[:, csx:csx + self.cropped_image_size,
                              csy:csy + self.cropped_image_size])
            for i, jpeg_string in enumerate(
                    self.jpeg_strings2['data'][start_id:end_id]):
                img = self.image.open(StringIO(jpeg_string))
                # center crop of image
                img = img.crop((csx, csy,
                                csx + self.cropped_image_size,
                                csy + self.cropped_image_size))
                if(img.mode != 'RGB'):
                    img = img.convert('RGB')
                self.inputs[:, i, np.newaxis] = (
                    np.transpose(np.array(
                        img, dtype=BDTYPE)[:, :, 0:3],
                        axes=[2, 0, 1]) - crop_mean_img).reshape((-1, 1))

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False

    # code below adapted from Alex Krizhevsky's cuda-convnet2 library,
    # make-data.py
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
        import scipy.io
        # manually reset number of cores to use if using multiprocessing
        pool_size = mp.cpu_count()
        os.system('taskset -cp 0-%d %s' % (pool_size, os.getpid()))

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
                for i in range(divup(len(l), partition_size))]

    def write_batches(self, target_dir, name, start_batch_num, labels,
                      jpeg_files):
        jpeg_files = self.partition_list(jpeg_files, self.output_batch_size)
        len_jpeg_files = len(jpeg_files)
        labels = self.partition_list(labels, self.output_batch_size)
        self.makedir(target_dir)
        logger.info("Writing %s batches (this can take hours)..." % name)
        for i, (labels_batch, jpeg_file_batch) in enumerate(
                zip(labels, jpeg_files)):
            t = time()
            jpeg_strings = self.resize_jpeg(
                [jpeg.read() for jpeg in jpeg_file_batch], as_string=True)
            batch_path = os.path.join(
                target_dir, '%s_batch_%d' % (name, start_batch_num + i))
            self.makedir(batch_path)
            # no subbatch support for now; do we really need them?
            # for j in range(0, len(labels_batch),
            # self.OUTPUT_SUB_BATCH_SIZE):
            j = 0
            my_pickle(os.path.join(batch_path, '%s_batch_%d.%d' %
                                   (name, start_batch_num + i, j /
                                    self.output_batch_size)),
                      {'data': jpeg_strings[j:j + self.output_batch_size],
                       'labels': labels_batch[j:j + self.output_batch_size]})
            logger.info("Wrote %s (%s batch %d of %d) (%.2f sec)" %
                        (batch_path, name, i + 1,
                         len_jpeg_files, time() - t))

            # multi-threaded image resize, not yet supported
            # proc = ResizeImages(self.output_image_size,
            #                     self.num_processes,
            #                     self.output_batch_size,
            #                     jpeg_file_batch,
            #                     i,
            #                     target_dir,
            #                     name,
            #                     start_batch_num,
            #                     labels_batch,
            #                     self.crop_to_square,
            #                     len_jpeg_files)
            # proc.start()
            # proc.join()

        return i + 1

# End of Krizhevsky code and Google Apache license

'''
Functions below are for multi-threaded image resize, not yet supported
'''


def resize_jpeg(jpeg_file_list, output_image_size, crop_to_square):
    from PIL import Image
    tgt = []
    print('called resize_jpeg')
    jpeg_strings = [jpeg.read() for jpeg in jpeg_file_list]
    for i, jpeg_string in enumerate(jpeg_strings):
        img = Image.open(StringIO(jpeg_string))

        # resize
        min_dim = np.min(img.size)
        scale_factor = np.float(output_image_size) / min_dim
        new_w = np.int(np.rint(scale_factor * img.size[0]))
        new_h = np.int(np.rint(scale_factor * img.size[1]))
        img = img.resize((new_w, new_h))  # todo: interp mode?

        # crop
        if crop_to_square:
            crop_start_x = (new_w - output_image_size) / 2
            crop_start_y = (new_h - output_image_size) / 2
            img = img.crop((crop_start_x, crop_start_y,
                            crop_start_x + output_image_size,
                            crop_start_y + output_image_size))
        else:
            raise NotImplementedError
        # return as string
        f = StringIO()
        img.save(f, "JPEG")
        tgt.append(f.getvalue())

    return tgt


def resize_jpeg_helper(args):
    return resize_jpeg(*args)


class ResizeImages(mp.Process):

    def __init__(self, output_image_size, num_processes, output_batch_size,
                 jpeg_file_batch, idx, target_dir, name, start_batch_num,
                 labels_batch, crop_to_square, len_jpeg_files):
        mp.Process.__init__(self)
        self.output_image_size = output_image_size
        self.num_processes = num_processes
        self.obs = output_batch_size
        self.jpeg_file_batch = jpeg_file_batch
        self.idx = idx
        self.target_dir = target_dir
        self.name = name
        self.start_batch_num = start_batch_num
        self.labels_batch = labels_batch
        self.crop_to_square = crop_to_square
        self.len_jpeg_files = len_jpeg_files

    def makedir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def run(self):
        t = time()
        start_idx = 0
        end_idx = self.obs

        start_list = range(
            start_idx, end_idx, self.obs / self.num_processes)
        end_list = range(start_idx + self.obs / self.num_processes,
                         end_idx + 1, self.obs / self.num_processes)
        logger.info('before loop')

        pool = mp.Pool(processes=self.num_processes)
        arglist = []
        for i in range(self.num_processes):
            file_list = self.jpeg_file_batch[start_list[i]:end_list[i]]
            arglist.append((file_list, self.output_image_size,
                            self.crop_to_square))
        results = pool.map(resize_jpeg_helper, arglist)
        logger.info('after map')

        # append the results together and return them
        tgt = []
        [tgt.extend(res) for res in results]

        batch_path = os.path.join(
            self.target_dir, '%s_batch_%d' % (self.name,
                                              self.start_batch_num + self.idx))
        self.makedir(batch_path)
        # no subbatch support for now; do we really need them?
        # for j in range(0, len(labels_batch),
        # self.OUTPUT_SUB_BATCH_SIZE):
        j = 0
        my_pickle(os.path.join(batch_path, '%s_batch_%d.%d' %
                               (self.name, self.start_batch_num + self.idx, j /
                                self.obs)),
                  {'data': tgt[j:j + self.obs],
                   'labels': self.labels_batch[j:j + self.obs]})
        logger.info("Wrote %s (%s batch %d of %d) (%.2f sec)" %
                    (batch_path, self.name, self.idx + 1,
                     self.len_jpeg_files, time() - t))
