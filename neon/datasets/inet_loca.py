"""
Provides neon.datasets.Dataset class for hurricane patches data
"""
import logging
import numpy as np
import h5py
import os #, sys
import xml.etree.ElementTree as ET
import glob
import urllib
from ipdb import set_trace as trace
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
        # scrape the data (takes a long time)
        if 0:
            self.scrape_and_crop()

        # now randomly crop a 64x64 patch to train on. If the image is too
        # small, just ignore it. Ideally this would be done for each epoch.
        cl = self.use_classes
        categories = glob.glob('/usr/local/data/inet/crop/n*')[0:cl] # enough is enough.
        one_hotness = len(categories) # number of categories we have.
        ims_per_class = []
        for r, d, f in os.walk('/usr/local/data/inet/crop'): ims_per_class += [len(f)]
        ims_per_class.pop(0)
        datasize = sum(ims_per_class[:cl])
        # perpare buffers
        inputs = np.zeros((datasize, 96*96*3))
        targets = np.zeros((datasize, one_hotness))

        i = 0 # data sample iterator
        for target, category in enumerate(categories):
            files = glob.glob(category+'/n*')
            one_hot = np.zeros(one_hotness)
            one_hot[target] = 1
            print target, "of", one_hotness, "with", ims_per_class[target]
            #sys.stdout.flush()
            for example in files:
                img = Image.open(example)
                if min(img.size)>100 and img.layers is 3: # it's big enough, crop it!
                    x_offset = int(np.random.rand() * (img.size[0]-96))
                    y_offset = int(np.random.rand() * (img.size[1]-96))
                    crop = img.crop((x_offset, y_offset, x_offset+96, y_offset+96)) # left, top, right, bottom
                    pixels = np.array(list(crop.getdata())).T.flatten()
                    #print "shape", pixels.T.shape, "into", inputs.shape, "at", i
                    inputs[i] = pixels.T
                    targets[i] = one_hot.T
                    i+=1

        # flatten into 2d array with rows as samples
        # and columns as features
        tr = self.training_size # from the yaml
        te = self.test_size
        print "tr+te", tr+te, "datasize", datasize
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
        #from matplotlib import pyplot as plt
        #import ipdb; ipdb.set_trace()
        map(normalize, [self.inputs['train'], self.inputs['validation']])

        # convert numpy arrays into CPUTensor backend
        self.format()
        # gives self.inputs['train'] list of 50 batches of size 27648, 100
        # and self.targets['train'] a list of 118 onehot x 100 batch size
        # list of 10 validation batches of same dimensions.



    def scrape_and_crop(self):
        """
        Download flicker images that have bounding boxes, and save to disk
        """
        basedir = '/usr/local/data/inet' # followed by n* directories
        url_file = os.path.join(basedir, 'fall11_urls.txt')
        targetdir = os.path.join(basedir, 'crop')

        # 1. convert the urls into a dict/hash. 1 Trillion x faster than list.
        dict_by_address = {}
        with open(url_file, 'r') as infile:
            for x, line in enumerate(infile):
                key, foo, value = line.rstrip().partition('\t')
                dict_by_address[key] = value
                if not x % 10000:
                    print "key", key, "value", value

        # 2. glob/walk directory and for each xml get the meta data
        base = 'n*' # all
        for outer_class in glob.glob(os.path.join(basedir,'Annotation',base)):
            subdir = outer_class.split('/')[-1]
            try:
                os.mkdir(os.path.join(targetdir,subdir))
            except OSError:
                pass
            try:
                os.mkdir(os.path.join(targetdir, '..', 'full', subdir))
            except OSError:
                pass
            print "------------ ENTERING CLASS", subdir, "-----------------"
            for xmlfile in glob.glob(os.path.join(outer_class, '*.xml')):
                # get the filename
                tree = ET.parse(xmlfile)
                root = tree.getroot()
                fnum = tree.find('filename').text
                jpg_path = os.path.join(targetdir, '..', 'full', subdir, fnum+'.jpg')
                # get the bounding box
                box=[]
                for i, bb in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                    box += [int(root[5][4][i].text)]
                # curl down the file:
                try:
                    url = dict_by_address[fnum]
                    print "hitting", fnum, url
                    # get the crop:
                    try:
                        urllib.urlretrieve(url, jpg_path)
                        img = Image.open(jpg_path)
                        #trace()
                        # TODO: assert img.size > box
                        if img.format is not 'JPEG':
                            print "FAIL! got png or gif or something"
                            os.unlink(jpg_path)
                        else:
                            crop = img.crop(box) # left, top, right, bottom
                            print "crop", img.size, "to", crop.size
                            crop.save(os.path.join(targetdir, subdir, fnum+'.jpg'), quality=95)
                    except:
                        print "404 error or worse"
                        try:
                            os.unlink(jpg_path)
                        except:
                            pass
                except:
                    print "key value pair not in fall 2011 dataset"

