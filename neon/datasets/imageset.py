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
import pycuda.driver as cuda
import sys
import imgworker
from time import time

logger = logging.getLogger(__name__)


class LoadDecodeThread(Thread):
    """
    Load and decode a macrobatch of images in a separate thread.

    Double-buffer macrobatch data structures. use ds.decode_buf_idx 


    """

    def __init__(self, ds):
        Thread.__init__(self)
        self.ds = ds


    def run(self):

        ds = self.ds
        bsz = ds.batch_size * ds.backend.par.size()
        jdict = ds.get_macro_batch()
        mac_sz = len(jdict['data'])
        b_idx = ds.decode_buf_idx


        # print "LoadDecodeThread start b_idx ", b_idx
        # ds.tgt_macro[b_idx] = jdict['targets'] if 'targets' in jdict else None
        # ds.lbl_macro = {k: jdict['labels'][k] for k in ds.label_list}

        # for lbl in ds.label_list:
        #     eye_buf = np.eye(ds.nclass)
        #     """
        #     need to understand the right data layout for transferring to gpu.
        #     """
        #     ds.lbl_macro1hot[b_idx][lbl] = eye_buf[ds.lbl_macro[b_idx][lbl]]

        # print ds.lbl_macro1hot[b_idx]
        imgworker.decode_list(jpglist=jdict['data'],
                              tgt=ds.uimg_macro[:mac_sz],
                              orig_size=ds.output_image_size,
                              crop_size=ds.cropped_image_size,
                              center=ds.predict, flip=True,
                              rgb=ds.rgb,
                              nthreads=ds.num_workers)
        mean_val = 127
        if ds.mean_norm:
            mean_val = ds.mean_crop.reshape((1, ds.npixels))
        np.subtract(ds.uimg_macro, mean_val, ds.img_macro)

        if mac_sz < ds.macro_size:
            ds.img_macro[mac_sz:] = 0

        ds.img_macroT[b_idx][:] = ds.img_macro.T.astype(ds.backend_type, order='C')
        for lbl in ds.label_list:
            ds.lbl_macroT[b_idx][lbl][:] = np.eye(ds.nclass[lbl])[jdict['labels'][lbl]].T.astype(ds.backend_type, order='C')
        ds.minis_per_macro = mac_sz / bsz

        # print "LoadDecodeThread finish b_idx ", b_idx
        import sys
        sys.stdout.flush()

        return


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

        if self.backend_type is 'np.float16':
            self.backend_type = np.float16
        else:
            self.backend_type = np.float32
        logger.info("Imageset initialized with dtype %f", self.backend_type)
        req_param(self, ['cropped_image_size', 'output_image_size',
                         'imageset', 'save_dir', 'repo_path', 'macro_size'])

        opt_param(self, ['image_dir'], os.path.join(self.repo_path,
                                                    self.imageset))

        self.rgb = True if self.num_channels == 3 else False
        self.norm_factor = 128. if self.mean_norm else 256.
        self.loader_thread = None
        self.copy_stream = None

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
                         'train_mean', 'val_mean', 'labels_dict', 'nclass'])

    def get_macro_batch(self):
        self.macro_idx = (self.macro_idx + 1 - self.startb) \
            % self.nmacros + self.startb
        fname = os.path.join(self.save_dir,
                             'data_batch_{:d}'.format(self.macro_idx))
        return deserialize(os.path.expanduser(fname), verbose=False)

    def init_mini_batch_producer(self, batch_size, setname, predict=False):
        # local shortcuts
        sbaf = self.backend.allocate_fragment
        npaf = self.backend.alloc_host_mem
        btype = self.backend_type
        self.batches_generated = 0
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
        self.mean_be = sbaf((self.npixels, 1), np.float32)
        self.mean_be.copy_from(self.mean_crop.reshape(
            (self.npixels, 1)).astype(np.float32))
        self.batch_size = batch_size
        self.predict = predict
        self.minis_per_macro = self.macro_size / batch_size

        if self.macro_size % batch_size != 0:
            raise ValueError('self.macro_size not divisible by batch_size')

        self.macro_idx = self.endb
        self.mini_idx = 0
        # self.mini_idx = self.minis_per_macro - 1

        # Allocate space for the host and device copies of input

        self.decode_buf_idx = 0
        self.num_decode_buf = 2
        self.decoder_thread = None

        self.tgt_macro = [None for i in range(self.num_decode_buf)]
        # self.lbl_macro = [None for i in range(self.num_decode_buf)]
        self.lbl_macro1hot = [{} for i in range(self.num_decode_buf)]
        inp_macro_shape = (self.macro_size, self.npixels)
        inp_shape = (self.npixels, self.batch_size)
        # self.uimg_macroT = npaf(inp_macro_shape, dtype=np.uint8)

        self.uimg_macro = np.zeros(inp_macro_shape, dtype=np.uint8)
        self.img_macro = np.zeros(inp_macro_shape, dtype=np.int8)
        self.inp_be = [sbaf(inp_shape, dtype=btype) for i in range(2)]

        self.img_macroT = [npaf((self.npixels, self.macro_size), dtype=btype) for i in range(self.num_decode_buf)]
        self.lbl_macroT = [{lbl: npaf((self.nclass[lbl], self.macro_size), dtype=btype) for lbl in self.label_list} for i in range(self.num_decode_buf)]
        # Don't worry about the target buffer for now

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

    def device_transfer_async(self, batch_idx):

        b_idx = self.active_buf_idx
        s_idx = batch_idx * self.batch_size
        e_idx = s_idx + self.batch_size
        d_idx = (self.batches_generated + 1) % 2
        h_img = self.img_macroT[b_idx][:, s_idx:e_idx]
        h_lbl = {k: self.lbl_macroT[b_idx][k][:, s_idx:e_idx] for k in self.label_list}

        d_img = self.inp_be[d_idx]
        d_lbl = self.lbl_be[d_idx]

        backend.scatter_async(h_img, d_img, self.copy_stream)
        for lbl in h_lbl:
            backend.scatter_async(h_lbl[lbl], d_lbl[lbl], self.copy_stream)
        return

    def get_data_from_loader(self, batch_idx):

        """
        need to decode a macrobatch up front before starting minibatch processing

        want to have second macrobatch decode running in parallel with minibatch decoding

        can't start another macrobatch until minibatch idx == 0 again freeing up the dbl buf

        """
        # print "start_loader batch_idx", batch_idx
        if batch_idx == 0:

            if self.decoder_thread != None:
                #no-op unless minibatch loading finished faster than macrobatch in bg thread
                self.decoder_thread.join()
            else:
                # special case for first run through
                self.decoder_thread = LoadDecodeThread(self)
                self.decoder_thread.start()
                self.decoder_thread.join()

            # usual case for kicking off a background macrobatch thread
            self.active_buf_idx = self.decode_buf_idx
            self.decode_buf_idx = (self.decode_buf_idx + 1) % self.num_decode_buf
            self.decoder_thread = LoadDecodeThread(self)
            self.decoder_thread.start()

        next_batch_idx = (batch_idx + 1) % self.minis_per_macro
        if self.copy_stream is None:
            self.copy_stream = self.backend.create_stream()
            self.device_transfer_async(batch_idx)
            self.copy_stream.synchronize()
            self.batches_generated += 1
            self.device_transfer_async(next_batch_idx)
        else:
            self.copy_stream.synchronize()
            self.batches_generated += 1
            self.device_transfer_async(next_batch_idx)

        import sys
        sys.stdout.flush()

    def get_mini_batch(self, batch_idx):
        self.get_data_from_loader(self.mini_idx)
        self.mini_idx = (self.mini_idx + 1) % self.minis_per_macro
        d_idx = self.batches_generated % 2
        return self.inp_be[d_idx], self.tgt_be[d_idx], self.lbl_be[d_idx]

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False
