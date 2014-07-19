"""
MNIST is a handwritten digit image dataset.
More info at: http://yann.lecun.com/exdb/mnist/
"""

import gzip
import logging
import os
import struct
import urllib

import numpy

from mylearn.backends._numpy import Numpy
from mylearn.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class MNIST(Dataset):
    """
    Sets up an MNIST dataset.
    """

    raw_base_url = 'http://yann.lecun.com/exdb/mnist/'
    #TODO: fix py3 compatibility
    raw_train_input_gz = urllib.basejoin(raw_base_url,
                                         'train-images-idx3-ubyte.gz')
    raw_train_target_gz = urllib.basejoin(raw_base_url,
                                          'train-labels-idx1-ubyte.gz')
    raw_test_input_gz = urllib.basejoin(raw_base_url,
                                        't10k-images-idx3-ubyte.gz')
    raw_test_target_gz = urllib.basejoin(raw_base_url,
                                         't10k-labels-idx1-ubyte.gz')
    
    # use numpy as default backend
    backend = Numpy

    def __init__(self, **kwargs):
        """
        Creates a new MNIST instance.

        The following optional keyword args are supported (can be read from
        config file too):

        :param repo_path: where to locally host this dataset on disk
        :type repo_path: str or None
        :returns: new MNIST dataset instance
        :rtype: MNIST
        """
        self.__dict__.update(kwargs)


    def read_image_file(self, fname, dtype=None):
        """
        Carries out the actual reading of MNIST image files.
        """
        with open(fname, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>iiii', f.read(16))
            if magic != 2051:
                raise ValueError('invalid MNIST image file: ' + fname)
            array = numpy.fromfile(f, dtype='uint8').reshape((num_images,
                                                              rows, cols))
        if dtype is not None:
            dtype = numpy.dtype(dtype)
            array = array.astype(dtype)
            array /= 255.
        return array


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
        if self.inputs['train'] is None:
            if 'repo_path' in self.__dict__:
                save_dir = os.path.join(self.repo_path,
                                        self.__class__.__name__)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                train_idcs = range(60000)
                numpy.random.shuffle(train_idcs)
                if 'sample_pct' in self.__dict__:
                    if self.sample_pct > 1.0: self.sample_pct /= 100.0
                    train_idcs = train_idcs[0:int(60000 * self.sample_pct)]
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
                    logger.info('loading: %s' % name)
                    if 'images' in repo_file and 'train' in repo_file:
                        indat = self.read_image_file(repo_file, 'float32')
                        # flatten to 1D images
                        indat = indat.reshape((60000, 784))[train_idcs]
                        self.inputs['train'] = self.backend.Tensor(indat)
                    elif 'images' in repo_file and 't10k' in repo_file:
                        indat = self.read_image_file(repo_file, 'float32')
                        indat = indat.reshape((10000, 784))
                        self.inputs['test'] = self.backend.Tensor(indat)
                    elif 'labels' in repo_file and 'train' in repo_file:
                        indat = self.read_label_file(repo_file)[train_idcs]
                        # Prep a 1-hot label encoding
                        tmp = numpy.zeros((len(train_idcs), 10))
                        for col in range(10):
                            tmp[:, col] = indat == col
                        self.targets['train'] = self.backend.Tensor(tmp)
                    elif 'labels' in repo_file and 't10k' in repo_file:
                        indat = self.read_label_file(repo_file)
                        tmp = numpy.zeros((10000, 10))
                        for col in range(10):
                            tmp[:, col] = indat == col
                        self.targets['test'] = self.backend.Tensor(tmp)
                    else:
                        logger.error('problems loading: %s' % name)
            else:
                raise AttributeError('repo_path not specified in config')
                # TODO: try and download and read in directly?
