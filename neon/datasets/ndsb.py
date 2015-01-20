# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
This dataset contains about 30K images obtained from training data
provided as part of the National Data Science Bowl - 2014.
More info at: http://www.datasciencebowl.com/
"""

import logging
import numpy as np
import os
import zipfile
import glob

try:
    from skimage import io, transform
except ImportError:
    pass

from neon.datasets.dataset import Dataset
from neon.util.compat import range
# from neon.util.persist import deserialize


logger = logging.getLogger(__name__)


class NDSB(Dataset):

    """
    Sets up an NDSB dataset.

    Attributes:
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str): where to locally host this dataset on disk
    """

    def __init__(self, **kwargs):
        self.dist_flag = False
        self.macro_batched = False
        self.__dict__.update(kwargs)

    def fetch_dataset(self, rootdir, leafdir):
        datadir = os.path.join(rootdir, leafdir)
        if os.path.exists(datadir):
            return True

        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        repofile = os.path.join(rootdir, leafdir + '.zip')
        if not os.path.exists(repofile):
            logger.warning('Could not find %s', repofile)
            return False

        logger.info('Unzipping: %s', repofile)
        infile = zipfile.ZipFile(repofile)
        infile.extractall(rootdir)
        infile.close()
        return True

    def read_images(self, rootdir, leafdir, wildcard=''):
        logger.info('Reading images from %s', leafdir)
        if self.fetch_dataset(rootdir, leafdir) is False:
            return None, None, None
        dirs = glob.glob(os.path.join(rootdir, leafdir, wildcard))
        classind = 0
        imagecount = 0
        filetree = {}
        for dirname in dirs:
            filetree[classind] = []
            for walkresult in os.walk(dirname):
                for filename in walkresult[2]:
                    filetree[classind].append(os.path.join(dirname, filename))
                    imagecount += 1
            classind += 1
        imagesize = self.image_width * self.image_width
        nclasses = len(filetree)
        inputs = np.zeros((imagecount, imagesize), dtype=np.float32)
        targets = np.zeros((imagecount, nclasses), dtype=np.float32)
        imageind = 0
        for classind in range(nclasses):
            for filename in filetree[classind]:
                img = io.imread(filename, as_grey=True)
                img = transform.resize(img, (self.image_width,
                                             self.image_width))
                img = np.float32(img)
                # Invert the greyscale.
                img = 1.0 - img
                inputs[imageind][:] = img.ravel()
                targets[imageind, classind] = 1
                imageind += 1
        return inputs, targets, filetree

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' not in self.__dict__:
            raise AttributeError('repo_path not specified in config')

        rootdir = os.path.join(self.repo_path,
                               self.__class__.__name__)
        inputs, targets, filetree = self.read_images(rootdir, 'train', '*')
        traininds = []
        valinds = []
        start = 0
        # Split into training and validation sets with similar
        # class distributions.
        for key, subtree in filetree.iteritems():
            count = len(subtree)
            end = start + count
            subrange = np.arange(start, end)
            np.random.shuffle(subrange)
            mid = int(count * 0.7)
            traininds += list(subrange[:mid])
            valinds += list(subrange[mid:])
            start = end
        np.random.shuffle(traininds)
        self.inputs['train'] = inputs[traininds]
        self.targets['train'] = targets[traininds]
        self.inputs['validation'] = inputs[valinds]
        self.targets['validation'] = targets[valinds]

        if 'sample_pct' in self.__dict__:
            self.sample_training_data()

        # Do not process the test set yet.
        if False:
            inputs, targets, filetree = self.read_images(rootdir, 'test')
            self.inputs['test'] = inputs
            self.targets['test'] = targets
        self.format()
