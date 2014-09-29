"""
Datasets with fake data for testing purposes.
"""

import logging
import numpy as np

from mylearn.datasets.dataset import Dataset


logger = logging.getLogger(__name__)


class URAND(Dataset):
    """
    Sets up a synthetic uniformly random dataset.

    Attributes:
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data
    """

    def __init__(self, ntrain, ntest, nin, nout, **kwargs):
        self.__dict__.update(kwargs)
        self.ntrain = ntrain
        self.ntest = ntest
        self.nin = nin
        self.nout = nout
        np.random.seed(0)

    def load_data(self, shape):
        data = np.random.uniform(low=0.0, high=1.0, size=shape)
        labels = np.random.randint(low=0, high=self.nout, size=shape[0])
        onehot = np.zeros((len(labels), self.nout), dtype=np.float32)
        for col in range(self.nout):
            onehot[:, col] = (labels == col)
        return (self.backend.wrap(data), self.backend.wrap(onehot))

    def load(self):
        self.inputs['train'], self.targets['train'] = (
            self.load_data((self.ntrain, self.nin)))
        self.inputs['test'], self.targets['test'] = (
            self.load_data((self.ntrain, self.nin)))
