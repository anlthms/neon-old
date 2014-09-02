"""
CIFAR-10 contains color images of 10 classes.
More info at: http://www.cs.toronto.edu/~kriz/cifar.html
"""

import tarfile
import logging
import os
import cPickle
import numpy as np

from mylearn.util.compat import PY3

from mylearn.datasets.dataset import Dataset

if PY3:
    from urllib.parse import urljoin as basejoin
else:
    from urllib import basejoin

logger = logging.getLogger(__name__)


class CIFAR10(Dataset):
    """
    Sets up a CIFAR-10 dataset.

    Attributes:
        url (str): where to find the source data
        backend (mylearn.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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

    def load_file(self, filename, nclasses):
        logger.info('loading: %s' % filename)
        fo = open(filename, 'rb')
        dict = cPickle.load(fo)
        fo.close()

        data = np.float32(dict['data'])
        data /= 255.
        labels = np.array(dict['labels'])
        onehot = np.zeros((len(labels), nclasses), dtype=np.float32)
        for col in range(nclasses):
            onehot[:, col] = (labels == col)
        return (data, onehot)

    def load(self):
        if self.inputs['train'] is None:
            if 'repo_path' in self.__dict__:
                ntrain_total = 50000
                nclasses = 10
                ncols = 32 * 32 * 3
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
