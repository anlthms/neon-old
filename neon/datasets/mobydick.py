# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
mobydick is a typewriterwritten book dataset.
More info at: http://www.gutenberg.org/ebooks/2701
"""

import logging
import numpy
import os

from neon.datasets.dataset import Dataset
from neon.util.compat import PY3, MPI_INSTALLED

logger = logging.getLogger(__name__)


class MOBYDICK(Dataset):

    """
    Sets up Moby Dick dataset.

    Attributes:
        raw_base_url (str): where to find the source data

        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """
    raw_base_url = 'http://www.gutenberg.org/cache/epub/2701/pg2701.txt'

    def __init__(self, **kwargs):
        self.dist_flag = False
        self.dist_mode = 0  # halo/tower method
        self.__dict__.update(kwargs)
        if self.dist_flag:
            if MPI_INSTALLED:
                from mpi4py import MPI
                self.comm = MPI.COMM_WORLD
                # for now require that comm.size is a square and divides 28
                if self.comm.size not in [1, 4, 16]:
                    raise AttributeError('MPI.COMM_WORLD.size not compatible')
            else:
                raise AttributeError("dist_flag set but mpi4py not installed")

    def read_txt_file(self, fname, dtype=None):
        """
        Carries out the actual reading
        """
        with open(fname, 'r') as f:
            text = f.read()
            numbers = numpy.fromstring(text, dtype='int8')
            onehots = numpy.zeros((128, numbers.shape[0]))
            for i in range(numbers.shape[0]):
                onehots[numbers[i], i] = 1

        if self.dist_flag:
            # leaving the container but no idea what to do here.
            pass
        else:
            array = onehots

        return array

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            train_idcs = range(1000000)  # 1M letters
            predict_idcs = range(self.unrolls, 1000000+self.unrolls)
            test_idcs = range(1000000, 1001000)
            testtarget_idcs = range(1000000+self.unrolls, 1001000+self.unrolls)
            if 'sample_pct' in self.__dict__:
                if self.sample_pct >= 1.0:
                    self.sample_pct /= 100.0
                    logger.info('sampling pct: %0.2f' % self.sample_pct)
                if self.sample_pct < 1.0:
                    # numpy.random.shuffle(train_idcs)
                    pass
                train_idcs = train_idcs[0:int(1000000 * self.sample_pct)]
                predict_idcs = predict_idcs[0:int(1000000 * self.sample_pct)]
            url = self.raw_base_url
            name = os.path.basename(url).rstrip('.txt')
            repo_file = os.path.join(save_dir, name + '.txt')
            if not os.path.exists(repo_file):
                self.download_to_repo(url, save_dir)
            logger.info('loading: %s' % name)
            indat = self.read_txt_file(repo_file, 'float32')
            self.inputs['train'] = indat[:, train_idcs].T
            self.targets['train'] = indat[:, predict_idcs].T
            self.inputs['test'] = indat[:, test_idcs].T
            self.targets['test'] = indat[:, testtarget_idcs].T

            self.format()
        else:
            raise AttributeError('repo_path not specified in config')
            # TODO: try and download and read in directly?
