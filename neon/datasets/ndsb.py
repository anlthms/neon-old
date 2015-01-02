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

from skimage import io, transform

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
            return False

        logger.info('unzipping: %s', repofile)
        infile = zipfile.ZipFile(repofile)
        infile.extractall(rootdir)
        infile.close()
        return True

    def copy_to_center(self, canvas, image):
        # Clip the image if it doesn't fit.
        if image.shape[0] > canvas.shape[0]:
            start = (image.shape[0] - canvas.shape[0]) / 2
            image = image[start:start + canvas.shape[0]]
        if image.shape[1] > canvas.shape[1]:
            start = (image.shape[1] - canvas.shape[1]) / 2
            image = image[:, start:start + canvas.shape[1]]

        ycenter = canvas.shape[0] / 2
        xcenter = canvas.shape[1] / 2
        yimage = ycenter - image.shape[0] / 2
        ximage = xcenter - image.shape[1] / 2
        canvas[yimage:yimage + image.shape[0],
               ximage:ximage + image.shape[1]] = image

    def read_images(self, rootdir, leafdir, wildcard=''):
        if self.fetch_dataset(rootdir, leafdir) is False:
            return None, None, None
        dirs = glob.glob(os.path.join(rootdir, leafdir, wildcard))
        classind = 0
        imagecount = 0
        filetree = {}
        for dirname in dirs:
            filetree[classind] = []
            logger.debug('walking', dirname)
            for walkresult in os.walk(dirname):
                for filename in walkresult[2]:
                    img = io.imread(os.path.join(dirname, filename),
                                    as_grey=True)
                    img = transform.resize(img, (self.image_width,
                                                 self.image_width))
                    img = np.float32(img)
                    # Invert the greyscale.
                    img = 1.0 - img
                    filetree[classind].append(img)
                    imagecount += 1
            classind += 1
        imagesize = self.image_width * self.image_width
        inputs = np.zeros((imagecount, imagesize), dtype=np.float32)
        imageind = 0
        for key, subtree in filetree.iteritems():
            for image in subtree:
                inputs[imageind][:] = image.ravel()
                imageind += 1
        return inputs, filetree, imagecount

    def read_targets(self, filetree, imagecount):
        nclasses = len(filetree)
        targets = np.zeros((imagecount, nclasses), dtype=np.float32)
        imageind = 0
        for classind in range(nclasses):
            for image in filetree[classind]:
                targets[imageind, classind] = 1
                imageind += 1
        return targets

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' not in self.__dict__:
            raise AttributeError('repo_path not specified in config')

        rootdir = os.path.join(self.repo_path,
                               self.__class__.__name__)
        inputs, filetree, imagecount = self.read_images(rootdir, 'train', '*')
        targets = self.read_targets(filetree, imagecount)

        inds = np.arange(inputs.shape[0])
        np.random.shuffle(inds)
        traincount = inputs.shape[0] * 0.7
        traincount -= traincount % 128
        self.inputs['train'] = inputs[inds[:traincount]]
        self.targets['train'] = targets[inds[:traincount]]

        if 'sample_pct' in self.__dict__:
            self.sample_training_data()

        endindex = inputs.shape[0]
        endindex -= endindex % 128
        self.inputs['validation'] = inputs[inds[traincount:endindex]]
        self.targets['validation'] = targets[inds[traincount:endindex]]

        inputs, filetree, imagecount = self.read_images(rootdir, 'test')
        self.inputs['test'] = inputs
        self.format()
