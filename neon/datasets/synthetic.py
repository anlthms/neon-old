"""
Datasets with fake data for testing purposes.
"""

import logging
import numpy as np

from PIL import Image, ImageDraw
from neon.datasets.dataset import Dataset


logger = logging.getLogger(__name__)


class UniformRandom(Dataset):
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
        return (data, onehot)

    def load(self):
        self.inputs['train'], self.targets['train'] = (
            self.load_data((self.ntrain, self.nin)))
        self.inputs['test'], self.targets['test'] = (
            self.load_data((self.ntest, self.nin)))
        self.format()


class ToyImages(Dataset):
    """
    Sets up a synthetic image classification dataset.

    Attributes:
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.ntrain = 128
        self.ntest = 128
        self.ifmheight = 32
        self.ifmwidth = self.ifmheight
        self.maxrad = self.ifmwidth / 2
        self.minrad = self.ifmwidth / 8
        self.nifm = 3
        self.nin = self.nifm * self.ifmheight * self.ifmwidth
        self.nout = 2
        assert self.ifmheight % 2 == 0
        assert self.ifmwidth % 2 == 0
        self.center = (self.ifmwidth / 2, self.ifmheight / 2)
        np.random.seed(0)

    def soa(self, aos, output):
        """
        Convert from array of structures to structure of arrays.
        """
        assert aos.ndim == 3
        routput = output.reshape((aos.shape[2], aos.shape[0], aos.shape[1]))
        for channel in xrange(aos.shape[2]):
            routput[channel] = aos[..., channel]

    def ellipse(self, draw, xrad, yrad):
        draw.rectangle((0, 0, self.ifmheight, self.ifmwidth),
                       fill=(0, 0, 0))
        xleft = self.center[0] - xrad
        yleft = self.center[1] - yrad
        xright = self.center[0] + xrad
        yright = self.center[1] + yrad
        red, green, blue = np.random.randint(0, 256, 3)
        draw.ellipse((xleft, yleft, xright, yright), fill=(red, green, blue))

    def circle(self, draw, rad):
        self.ellipse(draw, rad, rad)

    def load_data(self, shape):
        data = np.zeros(shape, dtype=np.float32)
        labels = np.zeros(shape[0], dtype=np.float32)
        im = Image.new('RGB', (self.ifmheight, self.ifmwidth))
        draw = ImageDraw.Draw(im)
        ncircles = shape[0] / 2

        for row in xrange(0, ncircles):
            rad = np.random.randint(self.minrad, self.maxrad)
            self.circle(draw, rad)
            self.soa(np.array(im), data[row])

        for row in xrange(ncircles, shape[0]):
            while True:
                xrad, yrad = np.random.randint(self.minrad, self.maxrad, 2)
                if xrad != yrad:
                    break
            self.ellipse(draw, xrad, yrad)
            labels[row] = 1
            self.soa(np.array(im), data[row])

        data /= 255
        onehot = np.zeros((len(labels), self.nout), dtype=np.float32)
        for col in xrange(self.nout):
            onehot[:, col] = (labels == col)
        return (data, onehot)

    def load(self):
        ntotal = self.ntrain + self.ntest
        inds = np.arange(ntotal)
        np.random.shuffle(inds)
        data, targets = self.load_data((ntotal, self.nin))
        self.inputs['train'] = data[inds[:self.ntrain]]
        self.targets['train'] = targets[inds[:self.ntrain]]
        self.inputs['test'] = data[inds[self.ntrain:]]
        self.targets['test'] = targets[inds[self.ntrain:]]
        self.format()
