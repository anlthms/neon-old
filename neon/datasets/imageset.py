# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic image-like dataset able to be processed in macro batches.
"""

import logging
import numpy as np
import os
import sys

from neon.datasets.dataset import Dataset
from neon.util.param import opt_param, req_param
from neon.util.persist import deserialize


logger = logging.getLogger(__name__)


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
        opt_param(self, ['dotransforms', 'square_crop'], False)
        opt_param(self, ['mean_norm', 'unit_norm'], False)

        opt_param(self, ['tdims'], 0)
        opt_param(self, ['label_list'], ['l_id'])
        opt_param(self, ['num_channels'], 3)

        opt_param(self, ['num_workers'], 6)
        opt_param(self, ['backend_type'], 'np.float32')

        self.__dict__.update(kwargs)

        if self.backend_type in ['float16', 'np.float16', 'numpy.float16']:
            self.backend_type = np.float16
        elif self.backend_type in ['float32', 'np.float32', 'numpy.float32']:
            self.backend_type = np.float32
        else:
            raise ValueError('Datatype not understood')
        logger.warning("Imageset initialized with dtype %s", self.backend_type)
        req_param(self, ['cropped_image_size', 'output_image_size',
                         'imageset', 'save_dir', 'repo_path', 'macro_size'])

        opt_param(self, ['image_dir'], os.path.join(self.repo_path,
                                                    self.imageset))

        self.rgb = True if self.num_channels == 3 else False
        self.norm_factor = 128. if self.mean_norm else 256.

    def load(self):
        bdir = os.path.expanduser(self.save_dir)
        cachefile = os.path.join(bdir, 'dataset_cache.pkl')
        if not os.path.exists(cachefile):
            logger.warning("Batch dir cache not found in %s:", cachefile)
            # response = 'Y'
            response = raw_input("Press Y to create, otherwise exit: ")
            if response == 'Y':
                from neon.util.batch_writer import (BatchWriter,
                                                    BatchWriterImagenet)

                if self.imageset.startswith('I1K'):
                    self.bw = BatchWriterImagenet(**self.__dict__)
                else:
                    self.bw = BatchWriter(**self.__dict__)
                self.bw.run()
                logger.warning('Done writing batches - please rerun to train.')
            else:
                logger.warning('Exiting...')
            sys.exit()
        cstats = deserialize(cachefile, verbose=False)
        if cstats['macro_size'] != self.macro_size:
            raise NotImplementedError("Cached macro size %d different from "
                                      "specified %d, delete save_dir %s "
                                      "and try again.",
                                      cstats['macro_size'],
                                      self.macro_size,
                                      self.save_dir)

        # Make sure only those properties not by yaml are updated
        cstats.update(self.__dict__)
        self.__dict__.update(cstats)
        req_param(self, ['ntrain', 'nval', 'train_start', 'val_start',
                         'train_mean', 'val_mean', 'labels_dict'])

    def get_macro_batch(self):
        self.macro_idx = (self.macro_idx + 1 - self.startb) \
            % self.nmacros + self.startb
        fname = os.path.join(self.save_dir,
                             'data_batch_{:d}'.format(self.macro_idx))
        return deserialize(os.path.expanduser(fname), verbose=False)

    def init_mini_batch_producer(self, batch_size, setname, predict=False):
        # local shortcuts
        sbe = self.backend.empty
        betype = self.backend_type
        sn = 'val' if (setname == 'validation') else setname
        osz = self.output_image_size
        csz = self.cropped_image_size
        self.npixels = csz * csz * self.num_channels

        self.startb = getattr(self, sn + '_start')
        self.nmacros = getattr(self, 'n' + sn)
        self.endb = self.startb + self.nmacros
        nrecs = self.macro_size * self.nmacros
        num_batches = int(np.ceil((nrecs + 0.0) / batch_size))
        self.mean_img = getattr(self, sn + '_mean')
        self.mean_img.shape = (self.num_channels, osz, osz)
        pad = (osz - csz) / 2
        self.mean_crop = self.mean_img[:, pad:(pad + csz), pad:(pad + csz)]
        self.mean_be = sbe((self.npixels, 1), dtype=betype)
        self.mean_be.copy_from(self.mean_crop.reshape(
            (self.npixels, 1)).astype(np.float32))

        self.batch_size = batch_size
        self.predict = predict
        self.minis_per_macro = self.macro_size / batch_size

        if self.macro_size % batch_size != 0:
            raise ValueError('self.macro_size not divisible by batch_size')

        self.macro_idx = self.endb
        self.mini_idx = self.minis_per_macro - 1

        # Allocate space for the host and device copies of input
        inp_macro_shape = (self.macro_size, self.npixels)
        inp_shape = (self.npixels, self.batch_size)
        self.img_macro = np.zeros(inp_macro_shape, dtype=np.uint8)
        self.inp_be = sbe(inp_shape, dtype=betype)

        # Allocate space for device side labels
        lbl_shape = (self.nclass, self.batch_size)
        self.lbl_be = {lbl: sbe(lbl_shape, dtype=betype)
                       for lbl in self.label_list}

        # Allocate space for device side targets if necessary
        tgt_shape = (self.tdims, self.batch_size)
        self.tgt_be = sbe(tgt_shape, dtype=betype) if self.tdims != 0 else None

        return num_batches

    def get_mini_batch(self, batch_idx):
        import imgworker

        # batch_idx is ignored
        betype = self.backend_type
        bsz = self.batch_size
        self.mini_idx = (self.mini_idx + 1) % self.minis_per_macro

        if self.mini_idx == 0:
            jdict = self.get_macro_batch()
            # This macro could be smaller than macro_size for last macro
            mac_sz = len(jdict['data'])
            self.tgt_macro = jdict['targets'] if 'targets' in jdict else None
            self.lbl_macro = {k: jdict['labels'][k] for k in self.label_list}

            imgworker.decode_list(jpglist=jdict['data'],
                                  tgt=self.img_macro[:mac_sz],
                                  orig_size=self.output_image_size,
                                  crop_size=self.cropped_image_size,
                                  center=self.predict, flip=True,
                                  rgb=self.rgb,
                                  nthreads=self.num_workers)
            if mac_sz < self.macro_size:
                self.img_macro[mac_sz:] = 0
            # Leave behind the partial minibatch
            self.minis_per_macro = mac_sz / bsz

        s_idx = self.mini_idx * bsz
        e_idx = (self.mini_idx + 1) * bsz

        # See if we are a partial minibatch
        self.inp_be.copy_from(
            self.img_macro[s_idx:e_idx].T.astype(betype, order='C'))

        if self.mean_norm:
            self.backend.subtract(self.inp_be, self.mean_be, self.inp_be)

        if self.unit_norm:
            self.backend.divide(self.inp_be, self.norm_factor, self.inp_be)

        for lbl in self.label_list:
            hl = np.squeeze(self.lbl_macro[lbl][s_idx:e_idx])
            one_hot_lbl = np.eye(self.nclass)[hl].T.astype(betype, order='C')
            self.lbl_be[lbl].copy_from(one_hot_lbl)

        if self.tgt_be is not None:
            self.tgt_be.copy_from(
                self.tgt_macro[:, s_idx:e_idx].astype(betype))

        return self.inp_be, self.tgt_be, self.lbl_be

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False