"""
Provides neon.datasets.Dataset class for hurricane patches data
"""
import logging
import numpy as np
import os
import xml.etree.ElementTree as ET
import glob
import urllib
from PIL import Image

from neon.datasets.dataset import Dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ImagenetLocalize(Dataset):
    """
    Playground to download images that have bounding boxes and pick out the
    best ones.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)  # this reads the yaml

    def load(self, sl=None):
        """
        {todo} Read data from h5 file, assume it's already been created.

        Labels are located at http://image-net.org/download-bboxes, i.e.
        http://image-net.org/Annotation/Annotation.tar.gz and I have them at
        /usr/local/data/inet/Annotation/n00007846/n00007846_6247.xml
        and the plan is to glob them, and to wget the jpeg.

        Part 2 is the list of file names from alex at
        /usr/local/data/inet/fall11_urls.txt
        """

        ws = self.window_size
        # scrape the data (takes a long time)
        if 0:
            self.scrape_and_crop()

        # now randomly crop a ws*ws patch to train on. If the image is too
        # small, just ignore it. Ideally this would be done for each epoch.
        cl = self.use_classes
        categories = glob.glob('/usr/local/data/inet/crop/n*')[0:cl]
        one_hotness = len(categories)  # number of categories we have.
        ims_per_class = []
        for r, d, f in os.walk('/usr/local/data/inet/crop'):
            ims_per_class += [len(f)]
        ims_per_class.pop(0)
        datasize = sum(ims_per_class[:cl])
        # perpare buffers
        inputs = np.zeros((datasize, ws*ws*3))
        targets = np.zeros((datasize, one_hotness))

        i = 0  # data sample iterator
        for target, category in enumerate(categories):
            files = glob.glob(category+'/n*')
            one_hot = np.zeros(one_hotness)
            one_hot[target] = 1
            logger.info("%d of %d with %d" % (target, one_hotness,
                        ims_per_class[target]))
            for example in files:
                img = Image.open(example)
                if (min(img.size) > 100) and (img.layers is 3):  # big enough?
                    x_offset = int(np.random.rand() * (img.size[0]-ws))
                    y_offset = int(np.random.rand() * (img.size[1]-ws))
                    crop = img.crop((x_offset, y_offset,
                                     x_offset+ws, y_offset+ws))  # l,top,r,bot
                    pixels = np.array(list(crop.getdata())).T.flatten()
                    inputs[i] = pixels.T
                    targets[i] = one_hot.T
                    i += 1

        # flatten into 2d array with rows as samples
        # and columns as features
        tr = self.training_size  # from the yaml
        te = self.test_size
        logger.info("train + test size %d and datasize %d" % (tr+te, datasize))
        assert tr+te < datasize

        self.inputs['train'] = inputs[:tr]
        self.targets['train'] = targets[:tr]

        # same with validation set
        self.inputs['validation'] = inputs[tr:tr+te]
        self.targets['validation'] = targets[tr:tr+te]

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

    def scrape_and_crop(self):
        """
        Download flicker images that have bounding boxes, and save to disk
        """
        basedir = '/usr/local/data/inet'  # followed by n* directories
        url_file = os.path.join(basedir, 'fall11_urls.txt')
        targetdir = os.path.join(basedir, 'crop')

        # 1. convert the urls into a dict/hash. 1 Trillion x faster than list.
        dict_by_address = {}
        with open(url_file, 'r') as infile:
            for x, line in enumerate(infile):
                key, foo, value = line.rstrip().partition('\t')
                dict_by_address[key] = value
                if not x % 10000:
                    logger.info("key %s value %s " % (key, value))

        # 2. glob/walk directory and for each xml get the meta data
        base = 'n*'  # all subdirectories start with n
        for outer_cls in glob.glob(os.path.join(basedir, 'Annotation', base)):
            subdir = outer_cls.split('/')[-1]
            try:
                os.mkdir(os.path.join(targetdir, subdir))
            except OSError:
                pass
            try:
                os.mkdir(os.path.join(targetdir, '..', 'full', subdir))
            except OSError:
                pass
            logger.debug("------------ ENTERING CLASS %s-----------------"
                         % subdir)
            for xmlfile in glob.glob(os.path.join(outer_cls, '*.xml')):
                # get the filename
                tree = ET.parse(xmlfile)
                root = tree.getroot()
                fnum = tree.find('filename').text
                jpg_path = os.path.join(targetdir, '..', 'full',
                                        subdir, fnum+'.jpg')
                # get the bounding box
                box = []
                for i, bb in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                    box += [int(root[5][4][i].text)]
                # curl down the file:
                try:
                    url = dict_by_address[fnum]
                    logger.debug("hitting %s %s" % (fnum, url))
                    # get the crop:
                    try:
                        urllib.urlretrieve(url, jpg_path)
                        img = Image.open(jpg_path)
                        # TODO: assert img.size > box
                        if img.format is not 'JPEG':
                            logger.warning("FAIL! got png or gif or something")
                            os.unlink(jpg_path)
                        else:
                            crop = img.crop(box)  # left, top, right, bottom
                            logger.info("crop %d to %d"
                                        % (img.size, crop.size))
                            crop.save(os.path.join(targetdir, subdir,
                                      fnum+'.jpg'), quality=95)
                    except:
                        logger.warning("404 error or worse")
                        try:
                            os.unlink(jpg_path)
                        except:
                            pass
                except:
                    logger.warning("key value pair not in fall 2011 dataset")
