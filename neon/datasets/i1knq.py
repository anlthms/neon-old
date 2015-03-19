# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
ImageNet 1k dataset
More information at: http://www.image-net.org/download-imageurls
Sign up for an ImageNet account to download the dataset!
"""

import logging
import os

import imgworker as iw
import numpy as np

from neon.datasets.dataset import Dataset
from neon.util.compat import range, pickle

logger = logging.getLogger(__name__)

# prefix for directory name where macro_batches are stored
prefix_macro = 'macro_batches_'


def my_pickle(filename, data):
    with open(filename, "w") as fo:
        pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)


def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = pickle.load(fo)
    fo.close()
    return contents


class I1Knq(Dataset):

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
        self.start_train_batch = -1
        self.end_train_batch = -1
        self.start_val_batch = -1
        self.end_val_batch = -1
        self.preprocess_done = False
        self.__dict__.update(kwargs)
        self.repo_path = os.path.expandvars(os.path.expanduser(self.repo_path))

        if not hasattr(self, 'save_dir'):
            self.save_dir = os.path.join(self.repo_path, 'I1K')

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
            self.save_dir = os.path.expandvars(os.path.expanduser(
                self.save_dir))
            save_dir = self.save_dir
            if os.path.exists(os.path.join(save_dir, prefix_macro + str(
                    self.output_image_size))):
                # delete load_dir if want to reload/reprocess dataset
                return

    def preprocess_images(self):
        # Depends on mean being saved to a cached file in the data directory
        # Otherwise will just subtract 128 uniformly
        osz = self.output_image_size
        csz = self.cropped_image_size
        logger.info("loading mean image")
        mean_path = os.path.join(self.save_dir, prefix_macro + str(osz),
                                 'i1kmean.pkl')

        try:
            self.mean_img = my_unpickle(mean_path)
            self.mean_img.shape = (3, osz, osz)
            pad = (osz - csz) / 2
            self.mean_crop = self.mean_img[:, pad:(pad + csz), pad:(pad + csz)]
            self.mean_be = self.backend.empty((self.npixels, 1))
            self.mean_be.copy_from(
                self.mean_crop.reshape(-1).astype(np.float32))

        except:
            logger.info("Unable to find mean img file, setting mean to 128.")
            self.mean_be = self.backend.empty((1, 1))
            self.mean_be[:] = 128.

        logger.info("done loading mean image")

    def get_macro_batch(self):
        self.macro_idx += 1
        if self.macro_idx > self.endb:
            self.macro_idx = self.startb
        batch_fname = '{}_batch_{:d}'.format(self.batch_type, self.macro_idx)
        batch_path = os.path.join(
            self.save_dir, prefix_macro + str(self.output_image_size),
            batch_fname)
        macro_fname = os.path.join(batch_path, batch_fname + '.0')
        return my_unpickle(macro_fname)

    def init_mini_batch_producer(self, batch_size, setname, predict=False):
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

        self.mini_idx = self.num_minibatches_in_macro - 1
        self.macro_idx = self.endb

        self.npixels = self.cropped_image_size * self.cropped_image_size * 3
        self.targets_macro = np.zeros((self.nclasses, self.output_batch_size),
                                      dtype=np.float32)
        self.img_macro = np.zeros(
            (self.output_batch_size, self.npixels), dtype=np.uint8)

        self.targets_be = self.backend.empty((self.nclasses, self.batch_size))

        self.inputs_be = self.backend.empty((self.npixels, self.batch_size))

        if not self.preprocess_done:
            self.preprocess_images()
            self.preprocess_done = True

        return num_batches

    def get_mini_batch(self, batch_idx):
        # batch_idx is ignored
        self.mini_idx += 1
        self.mini_idx = self.mini_idx % self.num_minibatches_in_macro
        if self.mini_idx == 0:
            self.jpeg_strings = self.get_macro_batch()

            labels = self.jpeg_strings['labels']
            labels = [item for sublist in labels for item in sublist]
            labels = np.asarray(labels, dtype=np.float32)
            for col in range(self.nclasses):
                self.targets_macro[col] = labels == col

            iw.decode_list(jpglist=self.jpeg_strings['data'],
                           tgt=self.img_macro,
                           orig_size=self.output_image_size,
                           crop_size=self.cropped_image_size,
                           center=self.predict, flip=True, nthreads=5)

        startidx = self.mini_idx * self.batch_size
        endidx = (self.mini_idx + 1) * self.batch_size

        # This is what we transfer over
        self.inputs_be.copy_from(
            self.img_macro[startidx:endidx].T.astype(np.float32, order='C'))
        self.targets_be.copy_from(
            self.targets_macro[:, startidx:endidx].astype(np.float32))

        self.backend.subtract(self.inputs_be, self.mean_be, self.inputs_be)

        return self.inputs_be, self.targets_be

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False
