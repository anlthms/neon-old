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

from skimage import io

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

    def fetch_dataset(self, save_dir):
        data_dir = os.path.join(save_dir, 'train')
        if not os.path.exists(data_dir):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            repo_file = os.path.join(save_dir, 'train.zip')
            assert os.path.exists(repo_file)

            logger.info('unzipping: %s', repo_file)
            infile = zipfile.ZipFile(repo_file)
            infile.extractall(save_dir)
            infile.close()

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

    def read_train_images(self, save_dir):
        dirs = glob.glob(os.path.join(save_dir, 'train', "*"))
        nclasses = len(dirs)
        filetree = {}
        classind = 0
        maxheight = 0
        maxwidth = 0
        sumheight = 0
        sumwidth = 0
        imagecount = 0
        for dirname in dirs:
            filetree[classind] = []
            logger.debug('walking', dirname)
            for walkresult in os.walk(dirname):
                for filename in walkresult[2]:
                    img = np.float32(io.imread(os.path.join(dirname, filename),
                                               as_grey=True))
                    # Invert the greyscale.
                    img = 255.0 - img
                    filetree[classind].append(img)
                    if img.shape[0] > maxheight:
                        maxheight = img.shape[0]
                    if img.shape[1] > maxwidth:
                        maxwidth = img.shape[1]
                    sumheight += img.shape[0]
                    sumwidth += img.shape[1]
                    imagecount += 1
            classind += 1

        logger.info('Mean height %d mean width %d max height %d max width %d',
                    sumheight / imagecount, sumwidth / imagecount,
                    maxheight, maxwidth)
        if maxheight > self.image_width or maxwidth > self.image_width:
            # The image width specified in the configuration file is too small.
            logger.warning('Clipping %dx%d images to %dx%d',
                           maxheight, maxwidth,
                           self.image_width, self.image_width)
        maxheight = self.image_width
        maxwidth = self.image_width
        inputs = np.zeros((imagecount, maxheight * maxwidth), dtype=np.float32)
        targets = np.zeros((imagecount, nclasses), dtype=np.float32)
        imageind = 0
        for classind in range(nclasses):
            for image in filetree[classind]:
                self.copy_to_center(
                    inputs[imageind].reshape(maxheight, maxwidth), image)
                targets[imageind, classind] = 1
                imageind += 1
        return inputs, targets

    def read_test_images(self, save_dir):
        dirname = os.path.join(save_dir, 'test')
        filetree = []
        maxheight = 0
        maxwidth = 0
        sumheight = 0
        sumwidth = 0
        imagecount = 0
        for walkresult in os.walk(dirname):
            for filename in walkresult[2]:
                img = np.float32(io.imread(os.path.join(dirname, filename),
                                           as_grey=True))
                # Invert the greyscale.
                img = 255.0 - img
                filetree.append(img)
                if img.shape[0] > maxheight:
                    maxheight = img.shape[0]
                if img.shape[1] > maxwidth:
                    maxwidth = img.shape[1]
                sumheight += img.shape[0]
                sumwidth += img.shape[1]
                imagecount += 1

        logger.info('Mean height %d mean width %d max height %d max width %d',
                    sumheight / imagecount, sumwidth / imagecount,
                    maxheight, maxwidth)
        if maxheight > self.image_width or maxwidth > self.image_width:
            # The image width specified in the configuration file is too small.
            logger.warning('Clipping %dx%d images to %dx%d',
                           maxheight, maxwidth,
                           self.image_width, self.image_width)
        maxheight = self.image_width
        maxwidth = self.image_width
        inputs = np.zeros((imagecount, maxheight * maxwidth), dtype=np.float32)
        targets = np.zeros((imagecount, 1), dtype=np.float32)
        imageind = 0
        for image in filetree:
            self.copy_to_center(
                inputs[imageind].reshape(maxheight, maxwidth), image)
            imageind += 1
        return inputs, targets

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' not in self.__dict__:
            raise AttributeError('repo_path not specified in config')

        save_dir = os.path.join(self.repo_path,
                                self.__class__.__name__)
        self.fetch_dataset(save_dir)
        inputs, targets = self.read_train_images(save_dir)
        inputs /= 255.

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

        inputs, targets = self.read_test_images(save_dir)
        inputs /= 255.
        self.inputs['test'] = inputs
        self.targets['test'] = targets

        self.format()
