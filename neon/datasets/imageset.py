# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""

"""

import logging
import gzip
import numpy as np
import os
from glob import glob
from time import time
from neon.datasets.dataset import Dataset
from neon.util.compat import range, pickle, StringIO
from neon.util.param import opt_param, req_param
import sys
import imgworker

from multiprocessing import Pool
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

TARGET_SIZE = 256
SQUARE_CROP = True


def my_pickle(filename, data):
    with open(filename, "w") as fo:
        pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)


def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = pickle.load(fo)
    fo.close()
    return contents


def proc_img(imgfile):
    im = Image.open(imgfile)

    # This part does the processing
    scale_factor = TARGET_SIZE / np.float32(min(im.size))
    (wnew, hnew) = map(lambda x: int(round(scale_factor * x)), im.size)

    if scale_factor != 1:
        filt = Image.BICUBIC if scale_factor > 1 else Image.ANTIALIAS
        im = im.resize((wnew, hnew), filt)

    if SQUARE_CROP is True:
        (cx, cy) = map(lambda x: (x - TARGET_SIZE) / 2, (wnew, hnew))
        im = im.crop((cx, cy, TARGET_SIZE, TARGET_SIZE))

    buf = StringIO()
    im.save(buf, format='JPEG')
    return buf.getvalue()


class BatchWriter(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.out_dir = os.path.expanduser(self.batch_dir)
        self.in_dir = os.path.expanduser(self.image_dir)
        self.batch_size = self.macro_batch_size
        global TARGET_SIZE, SQUARE_CROP
        TARGET_SIZE = self.output_image_size
        SQUARE_CROP = self.square_crop
        opt_param(self, ['validation_pct'], 0.2)
        self.train_file = os.path.join(self.out_dir, 'train_file.csv.gz')
        self.val_file = os.path.join(self.out_dir, 'val_file.csv.gz')
        self.stats = os.path.join(self.out_dir, 'dataset_cache.pkl')
        self.val_mean = np.zeros((self.output_image_size,
                                 self.output_image_size, 3), dtype=np.uint8)
        self.train_mean = np.zeros((self.output_image_size,
                                   self.output_image_size, 3), dtype=np.uint8)

    def __str__(self):
        pairs = map(lambda a: a[0] + ': ' + a[1],
                    zip(self.__dict__.keys(),
                        map(str, self.__dict__.values())))
        return "\n".join(pairs)

    def write_csv_files(self):
        posfiles = glob(os.path.join(self.in_dir, '1', '*.JPEG'))
        negfiles = glob(os.path.join(self.in_dir, '0', '*.JPEG'))

        poslines = [(filename, 1, 0, 1) for filename in posfiles]
        neglines = [(filename, 0, 1, 0) for filename in negfiles]

        v_idxp = int(self.validation_pct * len(poslines))
        v_idxn = int(self.validation_pct * len(neglines))

        np.random.shuffle(poslines)
        np.random.shuffle(neglines)

        tlines = poslines[v_idxp:] + neglines[v_idxn:]
        vlines = poslines[:v_idxp] + neglines[:v_idxn]

        np.random.shuffle(tlines)
        np.random.shuffle(vlines)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        for ff, ll in zip([self.train_file, self.val_file], [tlines, vlines]):
            with gzip.open(ff, 'wb') as f:
                f.write('filename,l_id,t0,t1\n')
                for tup in ll:
                    f.write('{},{},{},{}\n'.format(*tup))
            f.close()

        # Write out cached stats for this data
        self.ntrain = (len(tlines) + self.batch_size - 1) / self.batch_size
        self.nval = (len(vlines) + self.batch_size - 1) / self.batch_size
        self.train_start = 0
        self.val_start = 10 ** int(np.log10(self.ntrain * 10))

        my_pickle(self.stats, {'ntrain': self.ntrain,
                               'nval': self.nval,
                               'train_start': self.train_start,
                               'val_start': self.val_start,
                               'macro_size': self.batch_size,
                               'train_mean': self.train_mean,
                               'val_mean': self.val_mean})

    def parse_file_list(self, infile):
        compression = 'gzip' if infile.endswith('.gz') else None
        df = pd.read_csv(infile, compression=compression)

        lk = filter(lambda x: x.startswith('l'), df.keys())
        tk = filter(lambda x: x.startswith('t'), df.keys())

        labels = {ll: np.array(df[ll].values, np.int32) for ll in lk}
        targets = np.array(df[tk].values, np.float32) if len(tk) > 0 else None
        imfiles = df['filename'].values

        return imfiles, labels, targets

    def write_batches(self, name, start, labels, imfiles, targets=None):
        psz = self.batch_size
        osz = self.output_image_size
        npts = (len(imfiles) + psz - 1) / psz

        imfiles = [imfiles[i*psz: (i+1)*psz] for i in range(npts)]

        if targets is not None:
            targets = [targets[i*psz: (i+1)*psz].T.copy() for i in range(npts)]

        labels = [{k: v[i*psz: (i+1)*psz] for k, v in labels.iteritems()}
                  for i in range(npts)]

        accum_buf = np.zeros((osz, osz, 3), dtype=np.int32)
        batch_mean = np.zeros(self.accum.shape, dtype=np.uint8)
        print "Writing %s batches..." % name
        for i, jpeg_file_batch in enumerate(imfiles):
            t = time()
            pool = Pool(processes=self.num_workers)
            jpeg_strings = pool.map(proc_img, jpeg_file_batch)
            pool.close()
            targets_batch = None if targets is None else targets[i]
            labels_batch = labels[i]
            bfile = os.path.join(self.out_dir, 'data_batch_%d' % (start + i))
            my_pickle(bfile, {'data': jpeg_strings,
                              'labels': labels_batch,
                              'targets': targets_batch})
            print "Wrote to %s (%s batch %d of %d) (%.2f sec)" % (
                self.out_dir, name, i + 1, len(imfiles), time() - t)

            # get the means and accumulate
            imgworker.calc_batch_mean(jpglist=jpeg_strings, tgt=batch_mean,
                                      orig_size=osz, rgb=True, nthreads=5)

            # scale for the case where we have an undersized batch
            if len(jpeg_strings) < self.batch_size:
                batch_mean *= len(jpeg_strings) / self.batch_size
            accum_buf += batch_mean

        mean_buf = self.train_mean if name == 'train' else self.val_mean
        mean_buf[:] = accum_buf / len(imfiles)

    def run(self):
        self.write_csv_files()
        namelist = ['train', 'validation']
        filelist = [self.train_file, self.val_file]
        startlist = [self.train_start, self.val_start]
        for sname, fname, start in zip(namelist, filelist, startlist):
            print sname, fname, start
            if fname is not None and os.path.exists(fname):
                imgs, labels, targets = self.parse_file_list(fname)
                self.write_batches(sname, start, labels, imgs, targets)
            else:
                print 'Skipping {}, file missing'.format(sname)


class Imageset(Dataset):

    """
    Sets up a macro batched imageset dataset.

    Assumes you have the data already partitioned and in macrobatch format

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

        opt_param(self, ['preprocess_done', 'dist_flag'], False)
        opt_param(self, ['dotransforms', 'square_crop', 'zero_center'], False)
        opt_param(self, ['tdims'], 0)
        opt_param(self, ['label_list'], ['l_id'])
        opt_param(self, ['num_workers'], 6)
        opt_param(self, ['backend_type'], np.float32)

        self.__dict__.update(kwargs)
        req_param(self, ['cropped_image_size', 'output_image_size',
                         'image_dir', 'batch_dir', 'macro_size'])

        from PIL import Image
        self.imlib = Image
        self.idims = (self.cropped_image_size ** 2) * 3

    def load(self):
        bdir = os.path.expanduser(self.batch_dir)
        cachefile = os.path.join(bdir, 'dataset_cache.pkl')
        if not os.path.exists(cachefile):
            logger.info("Batch dir cache not found in %s:", cachefile)
            response = 'Y'
            # response = raw_input("Press Y to create, otherwise exit: ")
            if response == 'Y':
                self.bw = BatchWriter(**self.__dict__)
                self.bw.run()
            else:
                logger.info('Exiting...')
                sys.exit()
        cstats = my_unpickle(cachefile)
        if cstats['macro_size'] != self.macro_size:
            raise NotImplementedError("Cached macro size %d different from "
                                      "specified %d, delete batch_dir %s "
                                      "and try again.",
                                      cstats['macro_size'],
                                      self.macro_size,
                                      self.batch_dir)
        self.__dict__.update(cstats)
        # Make sure these properties are set by the cachefile
        req_param(self, ['ntrain', 'nval', 'train_start', 'val_start',
                         'train_mean', 'val_mean'])
        sys.exit()

    def get_macro_batch(self):
        self.macro_idx = (self.macro_idx + 1 - self.startb) \
                            % self.nmacros + self.startb
        fname = os.path.join(self.batch_dir,
                             'data_batch_{:d}'.format(self.macro_idx))
        return my_unpickle(os.path.expanduser(fname))

    def init_mini_batch_producer(self, batch_size, setname, predict=False):
        # local shortcuts
        sbe = self.backend.empty
        betype = self.backend_type
        sn = 'val' if (setname == 'validation') else setname
        osz = self.output_image_size
        csz = self.cropped_image_size

        self.startb = getattr(self, sn + '_start')
        self.nmacros = getattr(self, 'n' + sn)
        self.endb = self.startb + self.nmacros
        nrecs = self.macro_size * self.nmacros
        num_batches = int(np.ceil((nrecs + 0.0) / batch_size))
        self.mean_img = getattr(self, sn + '_mean')

        self.mean_img.shape = (3, osz, osz)
        pad = (osz - csz) / 2
        self.mean_crop = self.mean_img[:, pad:(pad + csz), pad:(pad + csz)]
        self.mean_be = sbe((self.npixels, 1))
        self.mean_be.copy_from(self.mean_crop.reshape(-1).astype(np.float32))

        self.batch_size = batch_size
        self.predict = predict
        self.minis_per_macro = self.macro_size / batch_size

        if self.macro_size % batch_size != 0:
            raise ValueError('self.macro_size not divisible by batch_size')

        self.macro_idx = self.endb
        self.mini_idx = self.minis_per_macro - 1
        self.npixels = self.cropped_image_size * self.cropped_image_size * 3

        # Allocate space for the host and device copies of input
        inp_macro_shape = (self.macro_size, self.npixels)
        inp_shape = (self.npixels, self.batch_size)
        self.img_macro = np.zeros(inp_macro_shape, dtype=np.uint8)
        self.inp_be = sbe(inp_shape, dtype=betype)

        # Allocate space for device side labels
        lbl_shape = (1, self.batch_size)
        self.lbl_be = {lbl: sbe(lbl_shape, dtype=betype)
                       for lbl in self.label_list}

        # Allocate space for device side targets if necessary
        tgt_shape = (self.tdims, self.batch_size)
        self.tgt_be = sbe(tgt_shape, dtype=betype) if self.tdims != 0 else None

        return num_batches

    def get_mini_batch(self, batch_idx):
        # batch_idx is ignored
        betype = self.backend_type
        self.mini_idx = (self.mini_idx + 1) % self.minis_per_macro

        if self.mini_idx == 0:
            jdict = self.get_macro_batch()
            self.tgt_macro = jdict['targets'] if 'targets' in jdict else None
            self.lbl_macro = {k: jdict['labels'][k] for k in self.label_list}

            imgworker.decode_list(jpglist=jdict['data'], tgt=self.img_macro,
                                  orig_size=self.output_image_size,
                                  crop_size=self.cropped_image_size,
                                  center=self.predict, flip=True, nthreads=5)

        s_idx = self.mini_idx * self.batch_size
        e_idx = (self.mini_idx + 1) * self.batch_size

        self.inp_be.copy_from(
            self.img_macro[s_idx:e_idx].T.astype(betype, order='C'))
        self.backend.subtract(self.inputs_be, self.mean_be, self.inputs_be)
        if self.zero_center:
            self.backend.divide(self.inputs_be, 128., self.inputs_be)

        for lbl in self.label_list:
            self.lbl_be[lbl].copy_from(
                self.lbl_macro[lbl][np.newaxis, s_idx:e_idx].astype(betype))

        if self.tgt_macro is not None:
            self.tgt_be.copy_from(
                self.tgt_macro[:, s_idx:e_idx].astype(betype))

        return self.inp_be, self.tgt_be, self.lbl_be

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False
