# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Toronto Face Dataset is a dataset of faces with labeled
identities and facial expressions.
http://aclab.ca/users/josh/TFD.html
"""

import logging
import os
import numpy as np

from neon.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class TFD(Dataset):

    """
    Sets up a TFD dataset.

    Attributes:
        train_input_48 (str): name of 48x48 train inputs
        train_input_96 (str): name of 96x96 train inputs
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        base_folder (str): where to find the source data
        fold (int): 0-4 fold of data to use
        image_size (int): 48 or 96 pixel images
    """
    set_map = {'unlabeled': 0, 'train': 1, 'valid': 2, 'test': 3}

    def __init__(self, base_folder, fold=0, image_size=48, **kwargs):
        self.inputs = {}
        self.targets = {}
        self.dist_flag = False
        self.num_test_sample = 10000
        self.macro_batched = False
        self.train_input_48 = 'TFD_48x48.mat'
        self.train_input_96 = 'TFD_96x96.mat'
        self.fold = fold
        self.__dict__.update(kwargs)
        if image_size == 48:
            self.train_input = os.path.join(base_folder, self.train_input_48)
        elif image_size == 96:
            self.train_input = os.path.join(base_folder, self.train_input_96)
        else:
            raise ValueError('image_size should be 48 or 96.')

    def initialize(self):
        # perform additional setup that can't be done at initial construction
        if self.dist_flag:
            self.comm = self.backend.comm
            if ((self.dist_mode in ['halopar', 'vecpar'] and self.comm.size
                 not in [1, 4, 16]) or
                (self.dist_mode == 'datapar' and self.batch_size
                 % self.comm.size)):
                raise AttributeError('MPI.COMM_WORLD.size not compatible')

    def adjust_for_dist(self):
        if not hasattr(self, 'comm'):
            self.initialize()
        comm_rank = self.comm.rank
        self.dist_indices = []
        img_width = self.image_size
        img_size = img_width ** 2

        if self.dist_mode == 'halopar':
            # this requires comm_per_dim to be a square for now
            self.comm_per_dim = int(np.sqrt(self.comm.size))
            self.px_per_dim = img_width / self.comm_per_dim
            r_i = []
            c_i = []
            # top left corner in 2-D image
            for row in range(self.comm_per_dim):
                for col in range(self.comm_per_dim):
                    r_i.append(row * self.px_per_dim)
                    c_i.append(col * self.px_per_dim)
            for r in range(r_i[comm_rank], r_i[comm_rank] + self.px_per_dim):
                self.dist_indices.extend(
                    [r * img_width + x for x in range(
                        c_i[comm_rank], c_i[comm_rank] + self.px_per_dim)])
        elif self.dist_mode == 'vecpar':
            start_idx = 0
            for j in range(comm_rank):
                start_idx += (img_size // self.comm.size +
                              (img_size % self.comm.size > j))
            nin = (img_size // self.comm.size +
                   (img_size % self.comm.size > comm_rank))
            self.dist_indices.extend(range(start_idx, start_idx + nin))
        elif self.dist_mode == 'datapar':
            # split into nr_nodes and nr_procs_per_node
            idcs_split = []
            self.num_procs = self.comm.size
            num_batches = len(self.train_idcs) / self.batch_size
            split_batch_size = self.batch_size / self.num_procs
            start_index = self.comm.rank * split_batch_size
            for j in range(num_batches):
                idcs_split.extend(self.train_idcs[start_index:start_index +
                                                  split_batch_size])
                start_index += self.batch_size
            self.train_idcs = idcs_split
            self.dist_indices = range(img_size)

            if self.num_test_sample % self.batch_size != 0:
                raise ValueError('num_test_sample mod dataset batch size != 0')

    def load(self):
        if self.inputs['train'] is not None:
            return
        from scipy.io import loadmat
        logger.info('loading: %s' % self.train_input)
        data = loadmat(self.train_input)
        for key in self.set_map.keys():
            set_indices = data['folds'][:, self.fold] == self.set_map[key]
            self.inputs[key] = data['images'][set_indices].astype('float32')
            if key != 'unlabeled':
                self.targets[key] = data['labs_ex'][set_indices]-1
                self.targets[key+'_id'] = data['labs_id'][set_indices]
