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
# import scipy.io #importing this breaks multiprocessing!
from random import shuffle
from time import time
from neon.datasets.dataset import Dataset
from neon.backends.gpu import GPU, GPUTensor
from neon.util.compat import MPI_INSTALLED, range
import sys
import threading
import Queue
import multiprocessing as mp
#from multiprocessing import sharedctypes
import shmarray

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

    def __init__(self, file_name_queue, macro_batch_queue):
        threading.Thread.__init__(self)
        # queue with file names of macro batches
        self.file_name_queue = file_name_queue
        # queue with file contents
        self.macro_batch_queue = macro_batch_queue

    def run(self):
        fname = self.file_name_queue.get(block=True)
        #logger.info('%s: loading file', fname)
        self.macro_batch_queue.put(my_unpickle(fname))
        #logger.info('%s: done loading file', fname)
        self.file_name_queue.task_done()
        if not macroq.empty():
            t = macroq.get()
            t.start()
            macroq.task_done()
        else:
            global macroq_flag
            macroq_flag = False

# for multi-threaded image resize, not yet supported
def resize_jpeg(jpeg_file_list, output_image_size, crop_to_square):
    tgt = []
    print 'called resize_jpeg'
    jpeg_strings = [jpeg.read() for jpeg in jpeg_file_list]
    print output_image_size, crop_to_square
    # else:
    #     # as numpy array, row order
    #     tgt = np.empty(
    #         (len(jpeg_strings), (output_image_size ** 2) * 3),
    #         dtype='float32')
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
        #return as string
        f = StringIO()
        img.save(f, "JPEG")
        tgt.append(f.getvalue())

    return tgt

# multi-threaded image resize, not yet supported
def resize_jpeg_helper(args):
    return resize_jpeg(*args)

# multi-threaded image resize, not yet supported
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
        #offset_list = range(0, self.obs, self.obs / self.num_processes)
        # print [(self.mb_id, x1, x2, x3) for (x1,x2,x3) in zip(start_list,
        # end_list, offset_list)]
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
        tgt=[]
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


class DecompressImages(threading.Thread):

    def __init__(self, mb_id, mini_batch_queue, batch_size, output_image_size,
                 jpeg_strings, targets_macro, backend, num_processes,
                 mean_img):
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        # mini-batch queue
        self.mini_batch_queue = mini_batch_queue
        self.batch_size = batch_size
        self.output_image_size = output_image_size
        self.jpeg_strings = jpeg_strings
        self.targets_macro = targets_macro
        self.backend = backend
        self.inputs = shmarray.create(
            ((self.output_image_size ** 2) * 3, self.batch_size),
            dtype='float32')
        self.num_processes = num_processes
        self.mean_img = mean_img

    def jpeg_decoder(self, start_id, end_id, offset):
        # convert jpeg string to numpy array
        #logger.info('jpeg decode start mb %d %d', start_id, offset)

        for i, jpeg_string in enumerate(
                self.jpeg_strings['data'][start_id:end_id]):
            img = Image.open(StringIO(jpeg_string))
            if img.mode == 'L':  # greyscale
                logger.debug('greyscale image found... tiling')
                self.inputs[:, i + offset, np.newaxis] = (np.tile(
                    np.array(img, dtype='float32').reshape((-1, 1)), (3, 1)) - 
                        self.mean_img)
            else:
                self.inputs[:, i + offset, np.newaxis] = (np.transpose(np.array(
                    img, dtype='float32')[:,:,0:3],
                        axes=[2, 0, 1]).reshape((-1, 1)) - self.mean_img)
        #logger.info('jpeg decode end mb %d %d', start_id, offset)

    def run(self):
        #logger.info('mini-batch decompress start %d', self.mb_id)
        # provide mini batch from macro batch
        start_idx = self.mb_id * self.batch_size
        end_idx = (self.mb_id + 1) * self.batch_size

        start_list = range(
            start_idx, end_idx, self.batch_size / self.num_processes)
        end_list = range(start_idx + self.batch_size / self.num_processes,
                         end_idx + 1, self.batch_size / self.num_processes)
        offset_list = range(
            0, self.batch_size, self.batch_size / self.num_processes)
        # print [(self.mb_id, x1, x2, x3) for (x1,x2,x3) in zip(start_list,
        # end_list, offset_list)]

        procs = [mp.Process(target=self.jpeg_decoder, args=[x1, x2, x3]) for (
            x1, x2, x3) in zip(start_list, end_list, offset_list)]
        for proc in procs:
            proc.daemon = True
            proc.start()

        targets = self.targets_macro[:, start_idx:end_idx].copy()
        #print 'DecompressImages', targets, np.sum(targets)
        [proc.join() for proc in procs]

        #logger.info('mini-batch decompress end %d', self.mb_id)

        self.mini_batch_queue.put([self.inputs, targets])
        if not miniq.empty():
            di = miniq.get()
            di.start()
            miniq.task_done()
        else:
            global miniq_flag
            miniq_flag = False

# ring buffer for GPU backend


class RingBuffer(object):

    def __init__(self, max_size, batch_size, num_targets, num_input_dims):
        self.max_size = max_size
        self.id = 0
        self.prev_id = 0
        tmp_input = np.empty((num_input_dims, batch_size), dtype='float32')
        tmp_target = np.empty((num_targets, batch_size), dtype='float32')

        self.inputs_backend = []
        self.targets_backend = []
        for i in range(max_size):
            self.inputs_backend.append(GPUTensor(tmp_input))
            self.targets_backend.append(GPUTensor(tmp_target))

    def add_item(self, inputs, targets, backend):
        self.prev_id = self.id
        #logger.info('start add_item')

        #tgt = np.argmax(targets,axis=0)
        #print 'additem', tgt, np.sum(tgt)
        # using ring buffer
        self.inputs_backend[self.id].set_host_mat(inputs)
        self.targets_backend[self.id].set_host_mat(targets)
        #logger.info('done with set_host_mat')
        #print "done with set_host_mat", time()
        self.inputs_backend[self.id].copy_to_device()
        #logger.info('done with input copy_to_device')
        #print "done with input copy_to_device", time()
        self.targets_backend[self.id].copy_to_device()

        #self.inputs_backend[self.id] = inputs
        #self.targets_backend[self.id] = targets

        #tgt = backend.empty((1, 100))
        #backend.argmax(self.targets_backend[self.id],
        #                           axis=0,
        #                               out=tgt)
        #print 'postcopy'
        #print self.targets_backend[self.id].asnumpyarray()
        
        #logger.info('end add_item')
        self.id += 1
        if self.id == self.max_size:
            self.id = 0


class GPUTransfer(threading.Thread):

    def __init__(self, mb_id, mini_batch_queue, gpu_queue, backend,
                 ring_buffer):
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        self.mini_batch_queue = mini_batch_queue
        self.gpu_queue = gpu_queue
        self.backend = backend
        self.ring_buffer = ring_buffer

    def run(self):
        #logger.info('backend mini-batch transfer start %d', self.mb_id)
        # threaded conversion of jpeg strings to numpy array
        # if self.mini_batch_queue.empty():
        #     logger.info("no item in mini batch queue for gpu transfer"
        #                        "waiting")
        inputs, targets = self.mini_batch_queue.get(block=True)

        #tgt = np.empty((1, 100))
        #tgt = np.argmax(targets,axis=0)
        #print 'gputransfer', tgt, tgt
        
        #logger.info("popped mini_batch_queue")

        if isinstance(self.backend, GPU):
            #logger.info("using GPU backend")
            self.ring_buffer.add_item(inputs, targets, self.backend)
            self.gpu_queue.put([self.ring_buffer.inputs_backend[
                                self.ring_buffer.prev_id],
                                self.ring_buffer.targets_backend[
                                    self.ring_buffer.prev_id]])
        else:
            inputs_backend = self.backend.array(inputs)
            targets_backend = self.backend.array(targets)
            self.gpu_queue.put([inputs_backend, targets_backend])
        #logger.info('backend mini-batch transfer done %d', self.mb_id)
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
        self.dist_flag = False
        self.macro_batched = False
        self.start_train_batch = -1
        self.end_train_batch = -1
        self.start_val_batch = -1
        self.end_val_batch = -1
        self.preprocess_done = False
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
            if self.start_train_batch != -1:
                # number of batches to train for this yaml file (<= total
                # available)
                self.n_train_batches = self.end_train_batch - \
                    self.start_train_batch + 1
            if self.start_val_batch != -1:
                # number of batches to validation for this yaml file (<= total
                # available)
                self.n_val_batches = self.end_val_batch - \
                    self.start_val_batch + 1

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            # todo handle dist case
            # if self.dist_flag:
            #    self.adjust_for_dist()

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
            #pickle the labels dic
            self.labels_dic = labels_dic

            with self.open_tar(ilsvrc_train_tar, 'training tar') as tf:
                synsets = tf.getmembers()
                synset_tars = [
                    tarfile.open(fileobj=tf.extractfile(s)) for s in synsets]
                # subsampling the first n tar files for now
                # todo: delete this line
                self.nclasses = self.max_tar_file #1000
                synset_tars = synset_tars[:self.max_tar_file]
                #self.nclasses = 1000
                
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
                # macro batch size
                print "total number of training files = ", len(train_jpeg_files)
                self.output_batch_size = 3072
                self.max_file_index = 3072*25
                jpeg_file_sample = train_jpeg_files[0:self.max_file_index]
                label_sample = train_labels[0:self.max_file_index]
                self.val_max_file_index = 3072*5
                # self.output_batch_size = 3072
                # self.max_file_index = 3072*26
                # jpeg_file_sample = train_jpeg_files[0:self.max_file_index]
                # label_sample = train_labels[0:self.max_file_index]

                # #todo: dbg
                flat_labels = [item for sublist in train_labels for item in sublist]
                flat_labels =  np.unique(flat_labels)
                print flat_labels
                for i in range(self.max_file_index):
                    for j in range(self.max_tar_file):
                        if label_sample[i][0] == flat_labels[j]:
                            label_sample[i] = [j]
                #self.val_max_file_index = 3072*6
                #val_label_sample = validation_labels[0:self.val_max_file_index]
                val_file_indices=[]
                val_label_sample = []
                for i in range(len(validation_labels)):
                    for j in range(self.max_tar_file):
                        if validation_labels[i][0] == flat_labels[j]:
                            val_file_indices.append(i)
                            val_label_sample.append([j])
                
                if self.macro_batched:
                    # Write training batches
                    self.num_train_macro_batches = self.write_batches(
                        os.path.join(save_dir, 'macro_batches5_'
                                     + str(self.output_image_size)),
                        'training', 0,
                        label_sample, jpeg_file_sample)
                    with self.open_tar(ilsvrc_validation_tar,
                                       'validation tar') as tf:
                        validation_jpeg_files = sorted(
                            [tf.extractfile(m) for m in tf.getmembers()],
                            key=lambda x: x.name)
                        # import pdb
                        # pdb.set_trace()
                        val_file_sample = [validation_jpeg_files[i] for i in val_file_indices]
                        #todo: shuffle the validation file indices and corresponding labels before subsampling?
                        #alternatively different macro / mini batch size for validation data
                        tmp_zip = zip(val_file_sample, val_label_sample)
                        shuffle(tmp_zip)
                        val_file_sample = [e[0] for e in tmp_zip]
                        val_file_sample = val_file_sample[0:self.val_max_file_index]
                        val_label_sample = [e[1] for e in tmp_zip]
                        val_label_sample = val_label_sample[0:self.val_max_file_index]
                        #val_file_sample = validation_jpeg_files[
                        #    0:self.val_max_file_index]
                        self.num_val_macro_batches = self.write_batches(
                            os.path.join(save_dir, 'macro_batches5_'
                                         + str(self.output_image_size)),
                            'validation', 0,
                            val_label_sample,
                            val_file_sample)
                    self.cur_train_macro_batch = 0
                    self.cur_train_mini_batch = 0
                    self.cur_val_macro_batch = 0
                    self.cur_val_mini_batch = 0
                # else:
                #     # one big batch (no macro-batches)
                #     # todo 1: resize in a multithreaded manner
                #     jpeg_mat = self.resize_jpeg(
                #         [jpeg.read() for jpeg in jpeg_file_sample],
                #         as_string=False)

                #     self.inputs['train'] = jpeg_mat
                #     # convert labels to one hot
                #     tmp = np.zeros(
                #         (self.nclasses, self.max_file_index), dtype='float32')
                #     for col in range(self.nclasses):
                #         tmp[col] = label_sample == col
                #     self.targets['train'] = tmp
                #     logger.info('done loading training data')

                #     with self.open_tar(ilsvrc_validation_tar,
                #                        'validation tar') as tf:
                #         validation_jpeg_files = sorted(
                #             [tf.extractfile(m) for m in tf.getmembers()],
                #             key=lambda x: x.name)
                #         val_file_sample = validation_jpeg_files[
                #             0:self.val_max_file_index]
                #         jpeg_mat = self.resize_jpeg(
                #             [jpeg.read() for jpeg in val_file_sample],
                #             as_string=False)

                #         self.inputs['test'] = jpeg_mat
                #         tmp = np.zeros(
                #             (self.nclasses, self.max_file_index),
                #             dtype='float32')
                #         for col in range(self.nclasses):
                #             tmp[col] = val_label_sample == col
                #         self.targets['test'] = tmp

                #     logger.info("done loading imagenet data")
                #     self.format()
        else:
            raise AttributeError('repo_path not specified in config')

    def resize_jpeg(self, jpeg_strings, as_string=False):
        # currently just does crop and resize, no bells and whistles of data
        # augmentation (no reflections, translations, etc.)
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

    def preprocess_images(self):
        # compute mean of all the images
        logger.info("preprocessing images (computing mean image)")
        self.mean_img = np.zeros(((self.output_image_size ** 2) * 3,1), dtype='float32')

        for i in range(self.n_train_batches):
            logger.info("preprocessing macro-batch %d :", i)
            batch_path = os.path.join(self.save_dir, 'macro_batches5_' + str(self.output_image_size),
                '%s_batch_%d' % ('training', i))
            j = 0
            file_path = os.path.join(batch_path, '%s_batch_%d.%d' % (
                        'training', i, j / self.output_batch_size))
            jpeg_strings = my_unpickle(file_path)
            for jpeg_string in jpeg_strings['data']:
                img = Image.open(StringIO(jpeg_string))
                if img.mode == 'L':  # greyscale
                    logger.debug('greyscale image found... tiling')
                    self.mean_img += np.tile(
                        np.array(img, dtype='float32').reshape((-1, 1)), (3, 1))
                else:
                    self.mean_img += np.transpose(np.array(
                        img, dtype='float32')[:,:,0:3],
                            axes=[2, 0, 1]).reshape((-1, 1))

        self.mean_img = self.mean_img/(self.n_train_batches*self.output_batch_size)
        logger.info("done preprocessing images (computing mean image)")

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

    def get_macro_batch(self, macro_batch_index):
        j = 0
        num_iter = 1
        if self.macro_batch_queue.empty():
            # using same buffer size as ring buffer for macro batch (host mem)
            num_iter = self.ring_buffer_size
            self.macro_batch_onque = macro_batch_index
        else:
            self.macro_batch_onque = self.get_next_macro_batch_id(
                self.batch_type, self.macro_batch_onque)

        for i in range(num_iter):
            if i > 0:
                self.macro_batch_onque = self.get_next_macro_batch_id(
                    self.batch_type, macro_batch_index)
            batch_path = os.path.join(
                self.save_dir, 'macro_batches5_' + str(self.output_image_size),
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
                macroq_flag = True

    def get_next_mini_batch_id(self, batch_type, mini_batch_index):
        next_mini_batch_id = mini_batch_index + 1
        if batch_type == 'training':
            if next_mini_batch_id >= self.num_minibatches_in_macro:
                next_mini_batch_id = 0
        elif batch_type == 'validation':
            if next_mini_batch_id >= self.num_minibatches_in_macro:
                next_mini_batch_id = 0
        return next_mini_batch_id

    def init_mini_batch_producer(self, batch_size, batch_type,
                                 raw_targets=False, ring_buffer_size=2):
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.raw_targets = raw_targets
        self.ring_buffer_size = ring_buffer_size
        if self.output_batch_size % batch_size != 0:
            raise ValueError('self.output_batch_size % batch_size != 0')
        else:
            self.num_minibatches_in_macro = self.output_batch_size / batch_size
        if isinstance(self.backend, GPU):
            if raw_targets:
                self.ring_buffer = RingBuffer(
                    max_size=ring_buffer_size,
                    batch_size=batch_size,
                    num_targets=1,
                    num_input_dims=(self.output_image_size ** 2) * 3)
            else:
                self.ring_buffer = RingBuffer(
                    max_size=ring_buffer_size,
                    batch_size=batch_size,
                    num_targets=self.nclasses,
                    num_input_dims=(self.output_image_size ** 2) * 3)
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
        
        #self.mean_img = np.zeros(((self.output_image_size ** 2) * 3,1), dtype='float32')
        if not self.preprocess_done:
            self.mean_img = np.zeros(((self.output_image_size ** 2) * 3,1), dtype='float32')
            self.preprocess_images()
            self.preprocess_done = True

    def get_mini_batch(self):
        # keep track of most recent batch and return the next batch
        if self.batch_type == 'training':
            cur_mini_batch_id = self.cur_train_mini_batch
            if cur_mini_batch_id == 0:
                # when cur_mini_batch is 0 enque a macro batch
                logger.info("train processing macro batch: %d",
                            self.cur_train_macro_batch)
                self.get_macro_batch(self.cur_train_macro_batch)
                self.cur_train_macro_batch = self.get_next_macro_batch_id(
                    'training', self.cur_train_macro_batch)
        elif self.batch_type == 'validation':
            cur_mini_batch_id = self.cur_val_mini_batch
            if cur_mini_batch_id == 0:
                logger.info(
                    "val processing macro batch: %d", self.cur_val_macro_batch)
                self.get_macro_batch(self.cur_val_macro_batch)
                self.cur_val_macro_batch = self.get_next_macro_batch_id(
                    'validation', self.cur_val_macro_batch)
        else:
            raise ValueError('Invalid batch_type in get_batch')

        num_iter_mini = 1
        num_iter_gpu = 1
        if cur_mini_batch_id == 0:
            # assuming at least num_iter mini batches in macro
            num_iter_mini = self.ring_buffer_size
            num_iter_gpu = self.ring_buffer_size
            self.mini_batch_onque = cur_mini_batch_id
            self.gpu_batch_onque = cur_mini_batch_id
        else:
            self.mini_batch_onque = self.get_next_mini_batch_id(
                self.batch_type, self.mini_batch_onque)
            self.gpu_batch_onque = self.get_next_mini_batch_id(
                self.batch_type, self.gpu_batch_onque)

        # deque next macro batch
        if self.mini_batch_onque == 0:
            try:
                self.macro_batch_queue.task_done()
            except:
                # allow for the first get from macro_batch_queue
                pass
            self.jpeg_strings = self.macro_batch_queue.get(block=True)
            # todo: this part could also be threaded for speedup
            # during run time extract labels
            labels = self.jpeg_strings['labels']
            #flatten the labels list of lists to a single list
            labels = [item for sublist in labels for item in sublist]
            labels = np.asarray(labels, dtype='float32')
            # import ipdb
            # ipdb.set_trace()
            #print np.sum(labels==0), np.sum(labels==1)

            if not self.raw_targets:
                self.targets_macro = np.zeros(
                    (self.nclasses, self.output_batch_size), dtype='float32')
                for col in range(self.nclasses):
                    self.targets_macro[col] = labels == col
            else:
                self.targets_macro = labels.reshape((1, -1))
                print self.targets_macro.shape

        for i in range(num_iter_mini):
            if i > 0:
                self.mini_batch_onque = self.get_next_mini_batch_id(
                    self.batch_type, self.mini_batch_onque)
            di = DecompressImages(self.mini_batch_onque, self.mini_batch_queue,
                                  self.batch_size, self.output_image_size,
                                  self.jpeg_strings, self.targets_macro,
                                  self.backend, self.num_processes,
                                  self.mean_img)
            di.setDaemon(True)
            global miniq_flag
            if miniq_flag:
                miniq.put(di)
            else:
                di.start()
                miniq_flag = True

        # todo:
        # test file reading speeds 3M, 30M, 90M, 300M
        # multiple MPI processes reading diff files from same disk

        # TODO: resize 256x256 image to 224x224 here

        # todo: run this with page locked memory
        # thread host -> gpu transfers onto a queue
        for i in range(num_iter_gpu):
            if i > 0:
                self.gpu_batch_onque = self.get_next_mini_batch_id(
                    self.batch_type, self.gpu_batch_onque)
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

        if self.batch_type == 'training':
            self.cur_train_mini_batch = self.get_next_mini_batch_id(
                self.batch_type, cur_mini_batch_id)
        elif self.batch_type == 'validation':
            self.cur_val_mini_batch = self.get_next_mini_batch_id(
                self.batch_type, cur_mini_batch_id)

        inputs_backend, targets_backend = self.gpu_queue.get(block=True)
        self.gpu_queue.task_done()
        
        # tgt = self.backend.empty((1, self.batch_size))
        # self.backend.argmax(targets_backend,
        #                            axis=0,
        #                                out=tgt)
        #print self.backend.sum(tgt)
        # import pdb
        # pdb.set_trace()
                    
        #print "returning from get_mini_batch: ", time()
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
        logger.info("Writing %s batches..." % name)
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
                          len(jpeg_files), time() - t))

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
