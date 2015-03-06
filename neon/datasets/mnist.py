# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
MNIST is a handwritten digit image dataset.
More info at: http://yann.lecun.com/exdb/mnist/
"""

import gzip
import logging
import numpy
import os
import struct
import numpy as np

from neon.datasets.dataset import Dataset
from neon.util.compat import PY3, range


if PY3:
    from urllib.parse import urljoin as basejoin
else:
    from urllib import basejoin

logger = logging.getLogger(__name__)


class MNIST(Dataset):

    """
    Sets up an MNIST dataset.

    Attributes:
        raw_base_url (str): where to find the source data
        raw_train_input_gz (str): URL of the full path to raw train inputs
        raw_train_target_gz (str): URL of the full path to raw train targets
        raw_test_input_gz (str): URL of the full path to raw test inputs
        raw_test_target_gz (str): URL of the full path to raw test targets
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    raw_base_url = 'http://yann.lecun.com/exdb/mnist/'
    raw_train_input_gz = basejoin(raw_base_url, 'train-images-idx3-ubyte.gz')
    raw_train_target_gz = basejoin(raw_base_url, 'train-labels-idx1-ubyte.gz')
    raw_test_input_gz = basejoin(raw_base_url, 't10k-images-idx3-ubyte.gz')
    raw_test_target_gz = basejoin(raw_base_url, 't10k-labels-idx1-ubyte.gz')

    def __init__(self, **kwargs):
        self.num_test_sample = 10000
        self.macro_batched = False
        self.__dict__.update(kwargs)

    def initialize(self):
        pass

    def read_image_file(self, fname, dtype=None):
        """
        Carries out the actual reading of MNIST image files.
        """
        with open(fname, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>iiii', f.read(16))
            if magic != 2051:
                raise ValueError('invalid MNIST image file: ' + fname)
            full_image = numpy.fromfile(f,
                                        dtype='uint8').reshape((num_images,
                                                                rows * cols))

        if dtype is not None:
            dtype = numpy.dtype(dtype)
            full_image = full_image.astype(dtype)
            full_image /= 255.

        return full_image

    def read_label_file(self, fname):
        """
        Carries out the actual reading of MNIST label files.
        """
        with open(fname, 'rb') as f:
            magic, num_labels = struct.unpack('>ii', f.read(8))
            if magic != 2049:
                raise ValueError('invalid MNIST label file:' + fname)
            array = numpy.fromfile(f, dtype='uint8')
        return array

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            self.repo_path = os.path.expandvars(os.path.expanduser(
                self.repo_path))
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for url in (self.raw_train_input_gz, self.raw_train_target_gz,
                        self.raw_test_input_gz, self.raw_test_target_gz):
                name = os.path.basename(url).rstrip('.gz')
                repo_gz_file = os.path.join(save_dir, name + '.gz')
                repo_file = repo_gz_file.rstrip('.gz')
                if not os.path.exists(repo_file):
                    self.download_to_repo(url, save_dir)
                    with gzip.open(repo_gz_file, 'rb') as infile:
                        with open(repo_file, 'w') as outfile:
                            for line in infile:
                                outfile.write(line)
                logger.info('loading: %s', name)
                if 'images' in repo_file and 'train' in repo_file:
                    indat = self.read_image_file(repo_file, 'float32')
                    # flatten to 1D images
                    self.inputs['train'] = indat
                elif 'images' in repo_file and 't10k' in repo_file:
                    indat = self.read_image_file(repo_file, 'float32')
                    self.inputs['test'] = indat[0:self.num_test_sample]
                elif 'labels' in repo_file and 'train' in repo_file:
                    indat = self.read_label_file(repo_file)
                    # Prep a 1-hot label encoding
                    tmp = numpy.zeros((indat.shape[0], 10))
                    for col in range(10):
                        tmp[:, col] = indat == col
                    self.targets['train'] = tmp
                elif 'labels' in repo_file and 't10k' in repo_file:
                    indat = self.read_label_file(
                        repo_file)[0:self.num_test_sample]
                    tmp = numpy.zeros((self.num_test_sample, 10))
                    for col in range(10):
                        tmp[:, col] = indat == col
                    self.targets['test'] = tmp
                else:
                    logger.error('problems loading: %s', name)
            if 'sample_pct' in self.__dict__:
                self.sample_training_data()
            self.format()
        else:
            raise AttributeError('repo_path not specified in config')
            # TODO: try and download and read in directly?
