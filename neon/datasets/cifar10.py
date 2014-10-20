"""
CIFAR-10 contains color images of 10 classes.
More info at: http://www.cs.toronto.edu/~kriz/cifar.html
"""

import tarfile
import logging
import os
import cPickle
import numpy as np
from mpi4py import MPI

from neon.datasets.dataset import Dataset


logger = logging.getLogger(__name__)


class CIFAR10(Dataset):

    """
    Sets up a CIFAR-10 dataset.

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
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self, **kwargs):
        self.dist_flag = False
        self.__dict__.update(kwargs)
        if self.dist_flag:
            self.comm = MPI.COMM_WORLD
            # for now require that comm.size is a square and divides 28
            if self.comm.size not in [1, 4, 16]:
                raise Exception('MPI.COMM_WORLD.size not compatible')

    def fetch_dataset(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        repo_gz_file = os.path.join(save_dir, os.path.basename(self.url))
        if not os.path.exists(repo_gz_file):
            self.download_to_repo(self.url, save_dir)

        data_file = os.path.join(save_dir, 'cifar-10-batches-py', 'test_batch')
        if not os.path.exists(data_file):
            logger.info('untarring: %s' % repo_gz_file)
            infile = tarfile.open(repo_gz_file)
            infile.extractall(save_dir)
            infile.close()

    def sample_training_data(self):
        if self.sample_pct != 100:
            train_idcs = np.arange(self.inputs['train'].shape[0])
            ntrain_actual = (self.inputs['train'].shape[0] *
                             int(self.sample_pct) / 100)
            np.random.shuffle(train_idcs)
            train_idcs = train_idcs[0:ntrain_actual]
            self.inputs['train'] = self.inputs['train'][train_idcs]
            self.targets['train'] = self.targets['train'][train_idcs]

    def adjust_for_dist(self):
        # computes the indices to load from input data for the dist case

        comm_rank = self.comm.rank
        # todo: will change for different x/y dims for comm_per_dim
        self.comm_per_dim = int(np.sqrt(self.comm.size))
        img_width = 32
        img_2d_size = img_width ** 2
        px_per_dim = img_width / self.comm_per_dim
        r_i = []
        c_i = []
        self.dist_indices = []
        # top left corner in 2-D image
        for row in range(self.comm_per_dim):
            for col in range(self.comm_per_dim):
                r_i.append(row * px_per_dim)
                c_i.append(col * px_per_dim)
        for ch in range(3):
            for r in range(r_i[comm_rank], r_i[comm_rank] + px_per_dim):
                self.dist_indices.extend(
                    [ch * img_2d_size + r * img_width + x for x in range(
                        c_i[comm_rank], c_i[comm_rank] + px_per_dim)])

    def load_file(self, filename, nclasses):
        logger.info('loading: %s' % filename)
        fo = open(filename, 'rb')
        dict = cPickle.load(fo)
        fo.close()

        full_image = np.float32(dict['data'])
        full_image /= 255.

        if self.dist_flag:
            # read corresponding 'quad'rant of the image
            data = full_image[:, self.dist_indices]
        else:
            data = full_image

        labels = np.array(dict['labels'])
        onehot = np.zeros((len(labels), nclasses), dtype=np.float32)
        for col in range(nclasses):
            onehot[:, col] = (labels == col)
        return (data, onehot)

    def load(self):
        if self.inputs['train'] is None:
            if 'repo_path' in self.__dict__:
                if self.dist_flag:
                    self.adjust_for_dist()
                    ncols = len(self.dist_indices)
                else:
                    ncols = 32 * 32 * 3

                ntrain_total = 50000
                nclasses = 10
                save_dir = os.path.join(self.repo_path,
                                        self.__class__.__name__)
                self.fetch_dataset(save_dir)
                self.inputs['train'] = self.backend.zeros((ntrain_total,
                                                          ncols),
                                                          dtype=np.float32)
                self.targets['train'] = self.backend.zeros((ntrain_total,
                                                           nclasses),
                                                           dtype=np.float32)
                for i in range(5):
                    filename = os.path.join(save_dir, 'cifar-10-batches-py',
                                            'data_batch_' + str(i + 1))
                    data, labels = self.load_file(filename, nclasses)
                    nrows = data.shape[0]
                    start = i * nrows
                    end = (i + 1) * nrows
                    self.inputs['train'][start:end] = data
                    self.targets['train'][start:end] = labels

                if 'sample_pct' in self.__dict__:
                    self.sample_training_data()

                filename = os.path.join(save_dir, 'cifar-10-batches-py',
                                        'test_batch')
                data, labels = self.load_file(filename, nclasses)
                self.inputs['test'] = self.backend.zeros((data.shape[0],
                                                          ncols),
                                                         dtype=np.float32)
                self.targets['test'] = self.backend.zeros((data.shape[0],
                                                           nclasses),
                                                          dtype=np.float32)
                self.inputs['test'][:] = data
                self.targets['test'][:] = labels
            else:
                raise AttributeError('repo_path not specified in config')
