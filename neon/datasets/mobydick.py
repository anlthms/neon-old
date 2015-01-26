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
from neon.util.compat import MPI_INSTALLED, range
from ipdb import set_trace as trace
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
        self.macro_batched = False
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
            assert(self.data_dim == 128), "one-hot encoded ASCII required"
            onehots = numpy.zeros((self.data_dim, numbers.shape[0]))
            for i in range(numbers.shape[0]):
                onehots[numbers[i], i] = 1

        if self.dist_flag:
            # leaving the container but no idea what to do here.
            pass
        else:
            array = onehots

        return array

    def transpose_batches(self, data):
            """
            Transpose each minibatch within the dataset.
            """
            bs = self.data_dim * self.unrolls
            dd = self.data_dim
            if data.shape[0] % bs != 0:
                logger.warning('Incompatible batch size. '
                               'Discarding %d samples...',
                               data.shape[0] % bs)
            nbatches = data.shape[0] / bs
            batchwise = [[] for k in range(nbatches)]
            for batch in range(nbatches):
                batchdata = [self.backend.array(data[batch * bs + k*dd :
                                                batch * bs + (k+1) * dd])
                             for k in range(self.unrolls)]
                batchwise[batch] = batchdata
            return batchwise

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            train_idcs = list(range(1000000))  # 1M letters out of 1.23M
            test_idcs = range(1000000, 1010000)
            if 'sample_pct' in self.__dict__:
                if self.sample_pct >= 1.0:
                    self.sample_pct /= 100.0
                    logger.info('sampling pct: %0.2f' % self.sample_pct)
                if self.sample_pct < 1.0:
                    # numpy.random.shuffle(train_idcs)
                    pass
                train_idcs = train_idcs[0:int(1000000 * self.sample_pct)]
            url = self.raw_base_url
            name = os.path.basename(url).rstrip('.txt')
            repo_file = os.path.join(save_dir, name + '.txt')
            if not os.path.exists(repo_file):
                self.download_to_repo(url, save_dir)
            logger.info('loading: %s' % name)
            indat = self.read_txt_file(repo_file, 'float32')

            self.preinputs = dict()
            self.preinputs['train'] = indat[:, train_idcs]
            self.preinputs['test'] = indat[:, test_idcs]

            for dataset in ('train', 'test'):
                num_batches = self.preinputs[dataset].shape[1]/self.batch_size
                idx_list = numpy.arange(num_batches * self.batch_size)
                idx_list = idx_list.reshape(self.batch_size, num_batches)
                splay_3d = self.preinputs[dataset][:, idx_list.T]
                splay_3d = numpy.transpose(splay_3d, (1, 0, 2))
                splay_3d = splay_3d.reshape(-1, self.batch_size)
                self.inputs[dataset] = splay_3d
                offbyone = numpy.zeros(splay_3d.shape)
                length = offbyone.shape[0]
                offbyone[0:length-128, :] = splay_3d[128:length, :]
                self.targets[dataset] = offbyone
            self.format()  # runs transpose_batches

        else:
            raise AttributeError('repo_path not specified in config')
            # TODO: try and download and read in directly?
