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
import matplotlib
import pylab
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from neon.datasets.dataset import Dataset
from neon.util.compat import range
from neon.util.persist import serialize, deserialize


logger = logging.getLogger(__name__)


SUBS = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
SESSIONS = range(1, 6)
FEATURES = range(1, 57)
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
        super(BCI, self).__init__(**kwargs)
        self.dist_flag = False
        self.macro_batched = False

    def fetch_dataset(self, setname, datadir):
        if os.path.exists(datadir):
            return True

        repofile = os.path.join(self.rootdir, setname + '.zip')
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

    def read_data(self, setname):
        pklpath = os.path.join(self.rootdir, 'inputdict.pkl')
        if os.path.exists(pklpath):
            return deserialize(pklpath)

        datadir = os.path.join(self.rootdir, setname)
        if self.fetch_dataset(setname, datadir) is False:
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

    def read_targets(self, subs, sessions):
        pklpath = os.path.join(self.rootdir, 'targetdict.pkl')
        if os.path.exists(pklpath):
            return deserialize(pklpath)
        targetfile = os.path.join(self.rootdir, 'TrainLabels.csv')
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

    def normalize(self, data):
        minval = np.min(data)
        data -= minval
        maxval = np.max(data)
        data /= maxval

    def prep_raw_data(self, inputdict, nrows, subs, sessions,
                      nsamples, features):
        nfeatures = len(features)
        inputs = np.zeros((nrows, nfeatures*nsamples))
        row = 0
        rinputs = inputs.reshape((nrows, nfeatures, nsamples))
        for subid in subs:
            for sessid in sessions:
                fbinds = np.nonzero(inputdict[subid][sessid][:, -1])[0]
                sessdata = inputdict[subid][sessid]
                for ind in fbinds:
                    for featind in range(nfeatures):
                        feat = features[featind]
                        rinputs[row, featind] = (
                            sessdata[ind:ind+nsamples, feat])
                    row += 1
        assert row == nrows
        return inputs

    def sgram(self, signal, height, width):
        sgram = pylab.specgram(signal, NFFT=128, Fs=2, noverlap=120,
                               cmap=matplotlib.cm.gist_heat)[0]
        return sgram[height, :width]

    def prep_sgram_data(self, inputdict, nrows, subs, sessions,
                        nsamples, features):
        rawinputs = self.prep_raw_data(
            inputdict, nrows, subs, sessions, nsamples, features)

        nfeatures = len(features)
        nrows = rawinputs.shape[0]
        height, width = 16, 16
        inputs = np.zeros((nrows, nfeatures, height, width))

        rawinputs = rawinputs.reshape((nrows, nfeatures, nsamples))
        for row in range(nrows):
            for featind in range(nfeatures):
                fvector = rawinputs[row, featind]
                sgram = self.sgram(fvector, height, width)
                inputs[row, featind] = sgram
        return inputs.reshape((nrows, nfeatures*height*width))

    def fft(self, data):
        return np.log10(np.absolute(np.fft.rfft(data, axis=-1)[..., 0:48]))

    def prep_eng_data(self, inputdict, nrows, subs, sessions,
                      nsamples, features):
        rawinputs = self.prep_raw_data(inputdict, nrows, subs, sessions,
                                       nsamples, features)
        nfeatures = len(features)
        nrows = rawinputs.shape[0]

        rawinputs = rawinputs.reshape((nrows, nfeatures, nsamples))
        fftres = self.fft(rawinputs)
        return fftres.reshape((nrows, np.prod(fftres.shape[1:])))

    def get_chan_locs(self, width):
        chanfile = os.path.join(self.rootdir, 'ChannelsLocation.csv')
        locs = np.genfromtxt(chanfile, delimiter=',', dtype='float32',
                             skip_header=1, usecols=[0, 2, 3])
        assert locs.shape[1] == 3
        nchannels = locs.shape[0]
        assert nchannels == 56
        cords = np.zeros((nchannels, 3))
        angs = locs[:, 2] * np.pi / 180
        rads = locs[:, 1]
        # X co-ordinate
        cords[:, 1] = rads * np.cos(angs)
        # Y co-ordinate
        cords[:, 2] = rads * np.sin(angs)
        cords /= np.max(cords)
        cords += 1
        cords /= 2
        cords *= width - 1
        cords = np.int32(np.round(cords))
        # The first column is the channel identifier (1-based index).
        cords[:, 0] = locs[:, 0]
        return cords

    def anim(self, vid):
        im = plt.imshow(vid[0])
        for frm in range(vid.shape[0]):
            im.set_data(vid[frm])
            plt.pause(0.08)
        plt.show()

    def convolve(self, vids, filtw):
        result = np.empty(vids.shape)
        nfrms = vids[0].shape[0]
        nrows = vids[0].shape[1]
        ncols = vids[0].shape[2]

        for samp in range(vids.shape[0]):
            for frm in range(nfrms):
                frm1 = frm - filtw
                frm1 = 0 if frm1 < 0 else frm1
                frm2 = frm + filtw
                frm2 = nfrms if frm2 > nfrms else frm2
                for row in range(nrows):
                    row1 = row - filtw
                    row1 = 0 if row1 < 0 else row1
                    row2 = row + filtw
                    row2 = nrows if row2 > nrows else row2
                    for col in range(ncols):
                        col1 = col - filtw
                        col1 = 0 if col1 < 0 else col1
                        col2 = col + filtw
                        col2 = ncols if col2 > ncols else col2
                        result[samp, frm, row, col] = vids[samp,
                                                           frm1:frm2,
                                                           row1:row2,
                                                           col1:col2].mean()
        vids[:] = result

    def convolve2(self, vids, filtw):
        result = np.empty(vids.shape)
        nfrms = vids[0].shape[0]
        nrows = vids[0].shape[1]
        ncols = vids[0].shape[2]

        for frm in range(nfrms):
            frm1 = frm - filtw
            frm1 = 0 if frm1 < 0 else frm1
            frm2 = frm + filtw + 1
            frm2 = nfrms if frm2 > nfrms else frm2
            for row in range(nrows):
                row1 = row - filtw
                row1 = 0 if row1 < 0 else row1
                row2 = row + filtw + 1
                row2 = nrows if row2 > nrows else row2
                for col in range(ncols):
                    col1 = col - filtw
                    col1 = 0 if col1 < 0 else col1
                    col2 = col + filtw + 1
                    col2 = ncols if col2 > ncols else col2
                    result[:, frm, row, col] = (
                        vids[:, frm1:frm2, row1:row2,
                             col1:col2].mean(axis=3).mean(axis=2).mean(axis=1))
        vids[:] = result

    def prep_vid_data(self, inputdict, nrows, subs, sessions,
                      nsamples, features):
        rawinputs = self.prep_raw_data(inputdict, nrows, subs, sessions,
                                       nsamples, features)
        nfeatures = len(features)
        nrows = rawinputs.shape[0]
        depth, height, width = 16, 16, 16
        stride = depth / 2
        vidstream = np.zeros((nrows, nsamples, height, width))
        rawinputs = rawinputs.reshape((nrows, nfeatures, nsamples))
        chanlocs = self.get_chan_locs(width)
        featind = 0
        for featind in range(nfeatures):
            ycord = chanlocs[featind, 2]
            xcord = chanlocs[featind, 1]
            vidstream[:, :, ycord, xcord] = rawinputs[:, featind]
        nclips = (nsamples - depth) / stride
        curfrm = 0
        inputs = np.zeros((nrows, nclips, depth, height, width))
        for clip in range(nclips):
            inputs[:, clip] = vidstream[:, curfrm:curfrm+depth]
            self.convolve(inputs[:, clip], 1)
            self.convolve(inputs[:, clip], 2)
            if False:
                self.anim(inputs[0, clip])
            curfrm += stride
        rinputs = inputs.reshape((nrows, np.prod(inputs.shape[1:])))
        return rinputs

    def load_set(self, inputdict, nrows, subs, name):
        savepath = os.path.join(self.rootdir, name + '.npy')
        if os.path.exists(savepath):
            logger.info('loading from %s', savepath)
            infile = open(savepath)
            inputs = np.load(infile)
            infile.close()
            return inputs
        inputs = self.prep_data(inputdict, nrows, subs, SESSIONS,
                                NSAMPLES, FEATURES)
        outfile = open(savepath, 'w')
        logger.info('saving to %s', savepath)
        np.save(outfile, inputs)
        outfile.close()
        return inputs

    def load_targets(self, targetdict, subs, sessions):
        nrows = 0
        for subid in subs:
            for sessid in sessions:
                nsessrows = targetdict[subid][sessid].shape[0]
                nrows += nsessrows

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
        return targets

    def rf(self, tinputs, ttargets, vinputs, vtargets):
        from sklearn.ensemble import RandomForestClassifier as rfc
        from sklearn import metrics
        model = rfc(n_estimators=200, verbose=0, n_jobs=8, random_state=1)
        model.fit(tinputs, ttargets)
        imp = np.array(model.feature_importances_)
        frank = np.zeros((len(imp), 2))

        frank[:, 0] = range(len(imp))
        frank[:, 1] = imp
        ord = np.argsort(imp)
        fsel = ord[::-1]
        frank = frank[fsel, :]
        print 'feature imp:', frank

        pred = model.predict_proba(vinputs)
        print 'rf auc:', metrics.roc_auc_score(vtargets[:, 0], pred[1][:, 0])

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' not in self.__dict__:
            raise AttributeError('repo_path not specified in config')

        self.rootdir = os.path.expanduser(
            os.path.join(self.repo_path, self.__class__.__name__))
        inputdict = self.read_data('train')
        subs = inputdict.keys()
        sessions = inputdict[subs[0]].keys()
        targetdict = self.read_targets(subs, sessions)

        subs = SUBS
        split = int(len(subs) * 0.6)
        np.random.seed(0)
        np.random.shuffle(subs)
        trainsubs = subs[:split]
        valsubs = subs[split:]

        if self.data_type == 'raw':
            self.prep_data = self.prep_raw_data
        elif self.data_type == 'sgram':
            self.prep_data = self.prep_sgram_data
        elif self.data_type == 'vid':
            self.prep_data = self.prep_vid_data
        elif self.data_type == 'eng':
            self.prep_data = self.prep_eng_data
        else:
            raise AttributeError('invalid data type specified')

        traintargets = self.load_targets(targetdict, trainsubs, SESSIONS)
        traininputs = self.load_set(inputdict, traintargets.shape[0],
                                    trainsubs, 'train-' + self.data_type)
        valtargets = self.load_targets(targetdict, valsubs, SESSIONS)
        valinputs = self.load_set(inputdict, valtargets.shape[0], valsubs,
                                  'validation-' + self.data_type)

        traininds = np.arange(traininputs.shape[0])
        np.random.shuffle(traininds)
        self.inputs['train'] = traininputs[traininds]
        self.targets['train'] = traintargets[traininds]
        self.inputs['validation'] = valinputs
        self.targets['validation'] = valtargets

        if 0:
            self.rf(traininputs[traininds], traintargets[traininds],
                    valinputs, valtargets)

        if 'sample_pct' in self.__dict__:
            self.sample_training_data()

        self.format()
