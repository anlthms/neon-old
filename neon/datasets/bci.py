# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
More info at: http://www.kaggle.com/c/inria-bci-challenge
"""

import logging
import numpy as np
import os
import zipfile

from neon.datasets.dataset import Dataset
from neon.util.compat import range
from neon.util.persist import serialize, deserialize


logger = logging.getLogger(__name__)


SUBS = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
SESSIONS = range(1, 6)
FEATURES = [15, 20, 31, 55]
NSAMPLES = 260


class BCI(Dataset):

    """
    Sets up an BCI dataset.

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

    def fetch_dataset(self, rootdir, setname, datadir):
        if os.path.exists(datadir):
            return True

        repofile = os.path.join(rootdir, setname + '.zip')
        if not os.path.exists(repofile):
            logger.warning('Could not find %s', repofile)
            return False

        if not os.path.exists(datadir):
            os.makedirs(datadir)
        logger.info('Unzipping: %s', repofile)
        infile = zipfile.ZipFile(repofile)
        infile.extractall(datadir)
        infile.close()
        return True

    def read_data(self, rootdir, setname):
        pklpath = os.path.join(rootdir, 'inputdict.pkl')
        if os.path.exists(pklpath):
            return deserialize(pklpath)

        datadir = os.path.join(rootdir, setname)
        if self.fetch_dataset(rootdir, setname, datadir) is False:
            return None

        logger.info('Reading data from %s', datadir)
        for walkresult in os.walk(datadir):
            filenames = walkresult[2]
            nfiles = len(filenames)
            inputdict = {}
            for filename in filenames:
                subid = int(filename.split('_')[1][1:])
                sessid = int(filename.split('_')[2].split('.')[0][-2:])
                if subid not in inputdict.keys():
                    inputdict[subid] = {}
                fullpath = os.path.join(datadir, filename)
                logger.info('Reading %s', fullpath)
                inputdict[subid][sessid] = np.genfromtxt(
                    fullpath, delimiter=',', dtype='float32', skip_header=1)
        serialize(inputdict, pklpath)
        return inputdict

    def read_targets(self, rootdir, subs, sessions):
        pklpath = os.path.join(rootdir, 'targetdict.pkl')
        if os.path.exists(pklpath):
            return deserialize(pklpath)
        targetfile = os.path.join(rootdir, 'TrainLabels.csv')
        rawtargets = np.genfromtxt(targetfile, dtype='S16,int',
                                   delimiter=',', skip_header=1)

        targets = np.zeros((rawtargets.shape[0], 3), dtype=np.float32)
        row = 0
        for item in rawtargets:
            subid = int(item[0].split('_')[0][1:])
            sessid = int(item[0].split('_')[1].split('.')[0][-2:])
            targets[row, 0] = subid
            targets[row, 1] = sessid
            # Indicates whether the prediction was good/bad.
            targets[row, 2] = item[1]
            row += 1
        assert row == rawtargets.shape[0]
        targetdict = {}
        for subid in subs:
            targetdict[subid] = {}
            for sessid in sessions:
                subtarget = targets[targets[:, 0] == subid]
                sesstarget = subtarget[subtarget[:, 1] == sessid]
                targetdict[subid][sessid] = sesstarget[:, 2]
        serialize(targetdict, pklpath)
        return targetdict

    def prep_data(self, inputdict, targetdict, subs, sessions,
                  nsamples, features):
        nfeatures = len(features)
        nrows = 0
        for subid in subs:
            for sessid in sessions:
                nsessrows = targetdict[subid][sessid].shape[0]
                assert nsessrows == (inputdict[subid][sessid][:, -1] == 1).sum()
                nrows += nsessrows

        inputs = np.zeros((nrows, nfeatures*nsamples))
        targets = np.zeros((nrows, 2))
        row = 0
        for subid in subs:
            for sessid in sessions:
                nsessrows = targetdict[subid][sessid].shape[0]
                for col in range(2):
                    tview = targets[row:row+nsessrows, col]
                    tview[targetdict[subid][sessid] == col] = 1
                row += nsessrows
        assert row == nrows
        row = 0
        for subid in subs:
            for sessid in sessions:
                fbinds = np.nonzero(inputdict[subid][sessid][:, -1])[0]
                sessdata = inputdict[subid][sessid]
                for ind in fbinds:
                    for gfind in range(nfeatures):
                        gf = features[gfind]
                        startcol = gfind * nsamples
                        endcol = startcol + nsamples
                        inputs[row, startcol:endcol] = (
                            sessdata[ind:ind+nsamples, gf])
                    row += 1
        assert row == nrows
        return inputs, targets

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' not in self.__dict__:
            raise AttributeError('repo_path not specified in config')

        rootdir = os.path.join(self.repo_path,
                               self.__class__.__name__)
        inputdict = self.read_data(rootdir, 'train')
        subs = inputdict.keys()
        sessions = inputdict[subs[0]].keys()
        targetdict = self.read_targets(rootdir, subs, sessions)

        subs = SUBS
        split = int(len(subs) * 0.6)
        np.random.seed(0)
        np.random.shuffle(subs)
        trainsubs = subs[:split]
        valsubs = subs[split:]

        traininputs, traintargets = self.prep_data(
            inputdict, targetdict, trainsubs,
            SESSIONS, NSAMPLES, FEATURES)
        valinputs, valtargets = self.prep_data(
            inputdict, targetdict, valsubs,
            SESSIONS, NSAMPLES, FEATURES)

        traininds = np.arange(traininputs.shape[0])
        np.random.shuffle(traininds)
        self.inputs['train'] = traininputs[traininds]
        self.targets['train'] = traintargets[traininds]
        self.inputs['validation'] = valinputs
        self.targets['validation'] = valtargets

        if 'sample_pct' in self.__dict__:
            self.sample_training_data()

        self.format()
