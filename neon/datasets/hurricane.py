"""
Provides neon.datasets.Dataset class for hurricane patches data
"""
import logging
import numpy as np
import h5py
import os

from neon.datasets.dataset import Dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Hurricane(Dataset):
    """
    Sets up the NERSC Mantissa hurricane dataset.

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if 'repo_path' not in kwargs:
            raise ValueError('Missing repo_path.')

        self.rootdir = os.path.join(self.repo_path, self.__class__.__name__)

    def load(self, sl=None):
        """
        Read data from h5 file, assume it's already been created.

        Create training and validation datasets from 1 or more
        prognostic variables.
        """
        f = h5py.File(os.path.join(self.rootdir, self.hdf5_file), 'r')

        one = f['1']
        zero = f['0']

        # [DEBUG] some debug settings
        # which variables to pick
        v = self.variables if 'variables' in self.__dict__ else range(8)
        cr = self.crop/2 if 'crop' in self.__dict__ else 32/2  # crop center
        tr = self.training_size
        te = self.test_size
        rng = slice(16-cr, 16+cr)  # thanks scott!

        # take equal number of hurricanes and non-hurricanes
        self.inputs['train'] = np.vstack((one[:tr, v, rng, rng],
                                          zero[:tr, v, rng, rng]))

        # one hot encoding required for MLP
        self.targets['train'] = np.vstack(([[1, 0]] * tr, [[0, 1]] * tr))

        # same with validation set
        self.inputs['validation'] = np.vstack((one[tr:tr+te, v, rng, rng],
                                              zero[tr:tr+te, v, rng, rng]))
        self.targets['validation'] = np.vstack(([[1, 0]] * te, [[0, 1]] * te))

        f.close()

        # flatten into 2d array with rows as samples
        # and columns as features
        dims = np.prod(self.inputs['train'].shape[1:])
        self.inputs['train'].shape = (2*tr, dims)
        self.inputs['validation'].shape = (2*te, dims)

        # shuffle training set
        s = range(len(self.inputs['train']))
        np.random.shuffle(s)
        self.inputs['train'] = self.inputs['train'][s]
        self.targets['train'] = self.targets['train'][s]

        def normalize(x):
            """Make each column mean zero, variance 1"""
            x -= np.mean(x, axis=0)
            x /= np.std(x, axis=0)

        map(normalize, [self.inputs['train'], self.inputs['validation']])

        # convert numpy arrays into CPUTensor backend
        self.format()
