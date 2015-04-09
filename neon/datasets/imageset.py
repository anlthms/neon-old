# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""

"""

import logging
import numpy as np
import os
from neon.datasets.dataset import Dataset
from neon.util.batch_writer import BatchWriter, BatchWriterImagenet
from neon.util.param import opt_param, req_param
from neon.util.persist import deserialize
from threading import Thread
import sys
import imgworker
from time import time

logger = logging.getLogger(__name__)


class DeviceTransferThread(Thread):
    def __init__(self, ds):
        Thread.__init__(self)
        self.ds = ds

    @staticmethod
    def transpose_and_transfer(data_in, data_out, backend):
        d_img, d_tgt, d_lbl = data_out
        h_img, h_tgt, h_lbl = data_in
        backend.scatter(h_img, d_img)
        for lbl in h_lbl:
            backend.scatter(h_lbl[lbl], d_lbl[lbl])
        if h_tgt is not None:
            backend.scatter(h_tgt, d_tgt)
        return

    def run(self):
        s_idx = self.ds.mini_idx * self.ds.batch_size
        e_idx = s_idx + self.ds.batch_size

        # Host versions of each var
        h_img = self.ds.img_macro[s_idx:e_idx]
        h_lbl = {k: self.ds.lbl_macro[k][s_idx:e_idx, np.newaxis]
                 for k in self.ds.label_list}
        h_tgt = None if self.ds.tgt_macro is None else self.ds.tgt_macro[s_idx:e_idx]

        # print "Putting into ", self.ds.d_idx, self.ds.n_idx, self.current_idx
        data_in = [h_img, h_tgt, h_lbl]
        data_out = self.ds.data[self.ds.d_idx]
        DeviceTransferThread.transpose_and_transfer(data_in, data_out,
                                                    self.ds.backend)

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
        opt_param(self, ['backend_type'], np.float32)

        self.__dict__.update(kwargs)
        req_param(self, ['cropped_image_size', 'output_image_size',
                         'imageset', 'save_dir', 'repo_path', 'macro_size'])

        opt_param(self, ['image_dir'], os.path.join(self.repo_path,
                                                    self.imageset))

        self.rgb = True if self.num_channels == 3 else False
        self.norm_factor = 128. if self.mean_norm else 256.
        self.loader_thread = None

    def load(self):
        bdir = os.path.expanduser(self.save_dir)
        cachefile = os.path.join(bdir, 'dataset_cache.pkl')
        if not os.path.exists(cachefile):
            logger.info("Batch dir cache not found in %s:", cachefile)
            # response = 'Y'
            response = raw_input("Press Y to create, otherwise exit: ")
            if response == 'Y':
                if self.imageset.startswith('I1K'):
                    self.bw = BatchWriterImagenet(**self.__dict__)
                else:
                    self.bw = BatchWriter(**self.__dict__)
                self.bw.run()
                logger.info('Done writing batches -- please rerun to train.')
            else:
                logger.info('Exiting...')
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
        sbaf = self.backend.allocate_fragment
        btype = self.backend_type
        self.batches_generated = -1
        sn = 'val' if (setname == 'validation') else setname
        osz = self.output_image_size
        csz = self.cropped_image_size
        self.npixels = csz * csz * self.num_channels

        self.startb = getattr(self, sn + '_start')
        self.nmacros = getattr(self, 'n' + sn)
        self.endb = self.startb + self.nmacros
        nrecs = self.macro_size * self.nmacros
        num_batches = int(np.ceil((nrecs + 0.0) / batch_size))
        num_batches /= self.backend.par.size()

        # This will be a uint8 matrix
        self.mean_img = getattr(self, sn + '_mean')
        self.mean_img.shape = (self.num_channels, osz, osz)
        pad = (osz - csz) / 2
        self.mean_crop = self.mean_img[:, pad:(pad + csz), pad:(pad + csz)]
        self.mean_be = sbaf((self.npixels, 1))
        self.mean_be.copy_from(self.mean_crop.reshape(
            (self.npixels, 1)).astype(np.float32))
        self.batch_size = batch_size
        self.predict = predict
        self.minis_per_macro = self.macro_size / batch_size

        if self.macro_size % batch_size != 0:
            raise ValueError('self.macro_size not divisible by batch_size')

        self.macro_idx = self.endb
        # self.mini_idx = 0
        self.mini_idx = self.minis_per_macro - 1

        # Allocate space for the host and device copies of input
        inp_macro_shape = (self.macro_size, self.npixels)
        inp_shape = (self.npixels, self.batch_size)
        self.uimg_macro = np.zeros(inp_macro_shape, dtype=np.uint8)
        self.img_macro = np.zeros(inp_macro_shape, dtype=np.int8)
        self.inp_be = [sbaf(inp_shape, dtype=btype) for i in range(2)]

        # Allocate space for device side labels
        lbl_shape = (1, self.batch_size)
        self.lbl_be = [{lbl: sbaf(lbl_shape, dtype=btype)
                       for lbl in self.label_list} for i in range(2)]

        # Allocate space for device side targets if necessary
        tgt_shape = (self.tdims, self.batch_size)
        self.tgt_be = [sbaf(tgt_shape, dtype=btype) for i in range(2)]

        self.data = [
            [self.inp_be[i], self.tgt_be[i], self.lbl_be[i]] for i in range(2)]
        self.d_idx = 0
        return num_batches

    def stage_next_mini_batch(self, batch_idx):
        # This is only run by the root process since it is staging data on the
        # host for distribution
        bsz = self.batch_size * self.backend.par.size()
        self.mini_idx = (self.mini_idx + 1) % self.minis_per_macro

        if self.backend.rank() != 0:
            return None, None, None

        if self.mini_idx == 0:
            # t = time()
            jdict = self.get_macro_batch()
            # logger.info("\tMacroBatch load time (%.4f sec):", time() - t)
            # This macro could be smaller than macro_size for last macro
            mac_sz = len(jdict['data'])
            self.tgt_macro = jdict['targets'] if 'targets' in jdict else None
            self.lbl_macro = {k: jdict['labels'][k] for k in self.label_list}
            # t = time()
            imgworker.decode_list(jpglist=jdict['data'],
                                  tgt=self.uimg_macro[:mac_sz],
                                  orig_size=self.output_image_size,
                                  crop_size=self.cropped_image_size,
                                  center=self.predict, flip=True,
                                  rgb=self.rgb,
                                  nthreads=self.num_workers)
            # logger.info("\tMacroBatch decode time (%.4f sec):", time() - t)
            # mean_val = 127
            # if self.mean_norm:
            #     mean_val = self.mean_crop.reshape((1, self.npixels))
            # np.subtract(self.uimg_macro, mean_val, self.img_macro)

            if mac_sz < self.macro_size:
                self.img_macro[mac_sz:] = 0
            # Leave behind the partial minibatch
            self.minis_per_macro = mac_sz / bsz

        s_idx = self.mini_idx * bsz
        e_idx = (self.mini_idx + 1) * bsz

        # Host versions of each var
        h_img = self.uimg_macro[s_idx:e_idx]
        h_lbl = {k: self.lbl_macro[k][s_idx:e_idx, np.newaxis]
                 for k in self.label_list}
        h_tgt = None if self.tgt_macro is None else self.tgt_macro[s_idx:e_idx]

        return h_img, h_tgt, h_lbl

    def transpose_and_transfer(self, data_in, data_out, backend):
        d_img, d_tgt, d_lbl = data_out
        h_img, h_tgt, h_lbl = data_in
        backend.scatter(h_img, d_img)
        for lbl in h_lbl:
            backend.scatter(h_lbl[lbl], d_lbl[lbl])
        if h_tgt is not None:
            backend.scatter(h_tgt, d_tgt)
        return

    def run(self):
        s_idx = self.mini_idx * self.batch_size
        e_idx = s_idx + self.batch_size

        # Host versions of each var
        h_img = self.img_macro[s_idx:e_idx]
        h_lbl = {k: self.lbl_macro[k][s_idx:e_idx, np.newaxis]
                 for k in self.label_list}
        h_tgt = None if self.tgt_macro is None else self.tgt_macro[s_idx:e_idx]

        # print "Putting into ", self.d_idx, self.n_idx, self.current_idx
        data_in = [h_img, h_tgt, h_lbl]
        data_out = self.data[self.d_idx]
        self.transpose_and_transfer(data_in, data_out, self.backend)

    def start_loader(self, batch_idx):
        bsz = self.batch_size * self.backend.par.size()
        if batch_idx == 0:
            jdict = self.get_macro_batch()
            mac_sz = len(jdict['data'])
            self.tgt_macro = jdict['targets'] if 'targets' in jdict else None
            self.lbl_macro = {k: jdict['labels'][k] for k in self.label_list}
            imgworker.decode_list(jpglist=jdict['data'],
                                  tgt=self.uimg_macro[:mac_sz],
                                  orig_size=self.output_image_size,
                                  crop_size=self.cropped_image_size,
                                  center=self.predict, flip=True,
                                  rgb=self.rgb,
                                  nthreads=self.num_workers)
            mean_val = 127
            if self.mean_norm:
                mean_val = self.mean_crop.reshape((1, self.npixels))
            np.subtract(self.uimg_macro, mean_val, self.img_macro)
            if mac_sz < self.macro_size:
                self.img_macro[mac_sz:] = 0
            self.minis_per_macro = mac_sz / bsz

        self.run()
        # self.loader_thread = DeviceTransferThread(self)
        # self.loader_thread.start()

    def get_data_from_loader(self):
        next_mini_idx = (self.mini_idx + 1) % self.minis_per_macro
        # self.d_idx = self.batches_generated % 2
        self.d_idx = 0
        self.n_idx = (self.batches_generated + 1) % 2

        self.start_loader(self.mini_idx)
        # if self.loader_thread is None:
        #     self.start_loader(self.mini_idx)
        #     self.loader_thread.join()
        #     self.start_loader(next_mini_idx)
        # else:
        #     self.loader_thread.join()
        #     if not self.loader_thread.is_alive():
        #         self.start_loader(next_mini_idx)

        self.batches_generated += 1
        self.mini_idx = next_mini_idx

    def get_mini_batch2(self, batch_idx):
        self.get_data_from_loader()
        return self.data[self.d_idx]

    def get_mini_batch(self, batch_idx):
        betype = self.backend_type
        bsz = self.batch_size * self.backend.par.size()
        s_idx = self.mini_idx * bsz
        e_idx = (self.mini_idx + 1) * bsz


        h_img, h_tgt, h_lbl = self.stage_next_mini_batch(batch_idx)
        # self.inp_be[0].copy_from(
        #     self.uimg_macro[s_idx:e_idx].T.astype(betype, order='C'))
        # self.inp_be[0].copy_from(
        #     h_img.T.astype(betype, order='C'))

        self.backend.scatter(self.uimg_macro[s_idx:e_idx], self.inp_be[0])
        if self.mean_norm:
            self.backend.subtract(self.inp_be[0], self.mean_be, self.inp_be[0])

        for lbl in self.label_list:
            self.backend.scatter(
                self.lbl_macro[lbl][s_idx:e_idx], self.lbl_be[0][lbl])
            # self.lbl_be[0][lbl].copy_from(
            #     self.lbl_macro[lbl][s_idx:e_idx].reshape((1,
            #                                               -1)).astype(betype))
        # for lbl in self.label_list:
        #     self.lbl_be[0][lbl].copy_from(
        #         h_lbl[lbl].reshape((1,-1)).astype(betype))
        # for lbl in self.label_list:
        #     hl = h_lbl[lbl] if h_lbl is not None else None
        #     self.backend.scatter(hl, self.lbl_be[0][lbl])
        if h_tgt is not None:
            self.backend.scatter(h_tgt, self.tgt_be[0])
        return self.inp_be[0], self.tgt_be[0], self.lbl_be[0]

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False
