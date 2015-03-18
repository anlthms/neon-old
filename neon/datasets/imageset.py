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
from neon.util.compat import range, pickle, queue, StringIO
from neon.util.param import opt_param, req_param
import threading
import sys
import imgworker

from multiprocessing import Pool
import pandas as pd
from PIL import Image
# importing scipy.io breaks multiprocessing! don't do it here!

logger = logging.getLogger(__name__)

# global queues to start threads
macroq = queue.Queue()
miniq = queue.Queue()
gpuq = queue.Queue()
macroq_flag = False
miniq_flag = False
gpuq_flag = False


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
        self.val_start = 10**int(np.log10(self.ntrain*10))

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
        nparts = (len(imfiles) + psz - 1) / psz

        imfiles = [imfiles[i*psz: (i+1)*psz] for i in range(nparts)]

        if targets is not None:
            targets = [targets[i*psz: (i+1)*psz] for i in range(nparts)]

        labels = [{k: v[i*psz: (i+1)*psz] for k, v in labels.iteritems()}
                  for i in range(nparts)]

        accum_buf = np.zeros((self.output_image_size,
                          self.output_image_size, 3), dtype=np.int32)
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
                                      orig_size=self.output_image_size,
                                      rgb=True, nthreads=5)

            # scale for the case where we have an undersized batch
            if len(jpeg_strings) < self.batch_size:
                batch_mean *= len(jpeg_strings) / self.batch_size
            accum_buf += batch_mean

        mean_buf = self.train_mean if name == 'train' else self.val_mean
        mean_buf[:] = accum / len(imfiles)

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


class LoadFile(threading.Thread):

    '''
    thread that handles loading the macrobatch from pickled file on disk
    '''

    def __init__(self, file_name, macro_batch_queue):
        threading.Thread.__init__(self)
        self.file_name = os.path.expanduser(file_name)
        # queue with file contents
        self.macro_batch_queue = macro_batch_queue

    def run(self):
        self.macro_batch_queue.put(my_unpickle(self.file_name))
        if not macroq.empty():
            t = macroq.get()
            t.start()
            macroq.task_done()
        else:
            global macroq_flag
            macroq_flag = False


class DecompressImages(threading.Thread):

    '''
    thread that decompresses/translates/crops/flips jpeg images on CPU in mini-
    batches
    '''

    def __init__(self, mb_id, mini_batch_queue, batch_size, output_image_size,
                 cropped_image_size, macro_data, mean_img, predict,
                 dotransforms=False):
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        # mini-batch queue
        self.mini_batch_queue = mini_batch_queue
        self.batch_size = batch_size
        self.output_image_size = output_image_size
        self.cropped_image_size = cropped_image_size
        self.img_macro = macro_data[0]['data']
        self.tgt_macro = macro_data[1]
        self.lbl_macro = macro_data[2]

        imsz = (cropped_image_size ** 2) * 3
        self.inputs = np.empty((batch_size, (cropped_image_size ** 2) * 3),
                               dtype=np.uint8)
        self.mean_img = mean_img
        self.tparams = {'center?': not predict,
                        'flip?': not predict and dotransforms}

    def run(self):
        # provide mini batch from macro batch
        s_idx = self.mb_id * self.batch_size
        e_idx = s_idx + self.batch_size

        imgworker.decode_list(jpglist=self.img_macro[s_idx:e_idx],
                              tgt=self.inputs,
                              orig_size=self.output_image_size,
                              crop_size=self.cropped_image_size,
                              center=self.tparams['center?'],
                              flip=self.tparams['flip?'],
                              rgb=True, calcmean=False, nthreads=5)

        targets = None
        if self.tgt_macro is not None:
            targets = self.tgt_macro[:, s_idx:e_idx].copy()

        labels = {k: np.array(
                  self.lbl_macro[k][np.newaxis, s_idx:e_idx].copy(),
                  dtype=np.float32)
                  for k in self.lbl_macro.keys()}

        logger.debug('mini-batch decompress end %d', self.mb_id)

        self.mini_batch_queue.put([self.inputs, targets, labels])
        if not miniq.empty():
            di = miniq.get()
            di.start()
            miniq.task_done()
        else:
            global miniq_flag
            miniq_flag = False


class RingBuffer(object):

    '''
    ring buffer for backend to assist in transferring data from host to gpu
    the buffers that live on host
    '''

    def __init__(self, max_size, batch_size, num_input_dims, num_tgt_dims,
                 label_list, backend):
        self.max_size = max_size
        self.id = self.prev_id = 0

        in_shape = (num_input_dims, batch_size)
        lbl_shape = (1, batch_size)
        tgt_shape = (num_tgt_dims, batch_size)

        self.inputs_be = [backend.empty(in_shape, dtype='float32')
                          for i in range(max_size)]

        self.labels_be = [{lbl: backend.empty(lbl_shape, dtype='float32')
                           for lbl in label_list} for i in range(max_size)]

        if num_tgt_dims is not None:
            self.targets_be = [backend.empty(tgt_shape, dtype='float32')
                               for i in range(max_size)]

    def add_item(self, inputs, targets, labels, backend):
        logger.debug('start add_item')

        # Handle transpose copy_from
        self.inputs_be[self.id].copy_from(inputs)

        for lbl in labels.keys():
            self.labels_be[self.id][lbl].copy_from(labels[lbl])

        if targets is not None:
            # XXX: temporary hack to format the targets.
            targets = targets.transpose().copy()
            self.targets_be[self.id].copy_from(targets)

        logger.debug('end add_item')

        self.prev_id = self.id
        self.id = (self.prev_id + 1) % self.max_size


class GPUTransfer(threading.Thread):

    '''
    thread to transfer data to GPU
    '''
    def __init__(self, mb_id, mini_batch_queue, gpu_queue, backend,
                 ring_buffer):
        threading.Thread.__init__(self)
        self.mb_id = mb_id
        self.mini_batch_queue = mini_batch_queue
        self.gpu_queue = gpu_queue
        self.backend = backend
        self.ring_buffer = ring_buffer
        logger.debug('GT %d created', self.mb_id)

    def run(self):
        from neon.backends.cpu import CPU
        logger.debug('backend mini-batch transfer start %d', self.mb_id)
        # threaded conversion of jpeg strings to numpy array
        # if no item in queue, wait
        inputs, targets, labels = self.mini_batch_queue.get(block=True)
        logger.debug("popped mini_batch_queue")

        if isinstance(self.backend, GPU):
            rbuf = self.ring_buffer
            rbuf.add_item(inputs, targets, labels, self.backend)
            self.gpu_queue.put([rbuf.inputs_be[rbuf.prev_id],
                                rbuf.targets_be[rbuf.prev_id],
                                rbuf.labels_be[rbuf.prev_id]])
        else:
            sbearray = self.backend.array
            inputs_be = sbearray(inputs)
            targets_be = None if targets is None else sbearray(targets)
            labels_be = sbearray(labels)
            self.gpu_queue.put([inputs_be, targets_be, labels_be])
        logger.debug('backend mini-batch transfer done %d', self.mb_id)
        self.mini_batch_queue.task_done()

        if not gpuq.empty():
            gt = gpuq.get()
            gt.start()
            try:
                gpuq.task_done()
            except:
                pass
        else:
            global gpuq_flag
            gpuq_flag = False
        logger.debug('Made it to the end of gpuqueue run for %d', self.mb_id)


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
        opt_param(self, ['tdims'], 0)
        opt_param(self, ['label_list'], ['l_id'])
        opt_param(self, ['num_workers'], 6)

        self.__dict__.update(kwargs)
        req_param(self, ['cropped_image_size', 'output_image_size',
                         'image_dir', 'batch_dir', 'macro_batch_size'])

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
        if cstats['macro_batch_size'] != self.macro_batch_size:
            raise NotImplementedError("Cached macro size %d different from "
                                      "specified %d, delete batch_dir %s "
                                      "and try again.",
                                      cstats['macro_batch_size'],
                                      self.macro_batch_size,
                                      self.batch_dir)
        self.__dict__.update(cstats)
        # Make sure these properties are set by the cachefile
        req_param(self, ['ntrain', 'nval', 'train_start', 'val_start',
                         'train_mean', 'val_mean'])
        sys.exit()

    def get_macro_batch(self):
        for i in range(self.num_iter_macro):
            self.macro_onque = (self.macro_onque + 1 - self.startb) \
                                % self.nmacros + self.startb
            fname = os.path.join(self.batch_dir,
                                 'data_batch_{:d}'.format(self.macro_onque))
            t = LoadFile(fname, self.macro_batch_queue)
            t.daemon = True
            global macroq_flag
            if macroq_flag:
                macroq.put(t)
            else:
                t.start()
                # if macroq is not empty then set macroq_flag to True
                macroq_flag = True
        self.num_iter_macro = 1

    def init_mini_batch_producer(self, batch_size, setname, predict=False):
        from neon.backends.cpu import CPU
        sn = 'val' if (setname == 'validation') else setname
        self.startb = getattr(self, sn + '_start')
        self.nmacros = getattr(self, 'n' + sn)
        self.endb = self.startb + self.nmacros
        nrecs = self.macro_batch_size * self.nmacros
        num_batches = int(np.ceil((nrecs + 0.0) / batch_size))
        self.mean_img = getattr(self, sn + '_mean')

        self.batch_size = batch_size
        self.predict = predict
        self.minis_per_macro = self.macro_batch_size / batch_size

        if self.macro_batch_size % batch_size != 0:
            raise ValueError('self.macro_batch_size % batch_size != 0')

        self.macro_onque = self.endb
        self.mini_onque = self.minis_per_macro - 1
        self.gpu_onque = self.minis_per_macro - 1
        self.num_iter_mini = self.ring_buffer_size
        self.num_iter_gpu = self.ring_buffer_size
        self.num_iter_macro = self.ring_buffer_size

        return num_batches

    def get_mini_batch(self, batch_idx):
        # threaded version of get_mini_batch
        # batch_idx is ignored
        logger.debug('\tPre Minibatch %d %d %d', batch_idx, self.num_iter_mini,
                     self.mini_onque)

        for i in range(self.num_iter_mini):
            self.mini_onque = (self.mini_onque + 1) % self.minis_per_macro
            if self.mini_onque == 0:
                self.get_macro_batch()
                # deque next macro batch
                try:
                    self.macro_batch_queue.task_done()
                except:
                    # allow for the first get from macro_batch_queue
                    pass
                self.jpeg_strings = self.macro_batch_queue.get(block=True)

            self.targets_macro = None
            if 'targets' in self.jpeg_strings:
                self.targets_macro = self.jpeg_strings['targets']

            self.labels_macro = {k: self.jpeg_strings['labels'][k]
                                 for k in self.jpeg_strings['labels'].keys()}

            macro_data = [self.jpeg_strings, self.targets_macro,
                          self.labels_macro]

            di = DecompressImages(self.mini_onque, self.mini_batch_queue,
                                  self.batch_size, self.output_image_size,
                                  self.cropped_image_size, macro_data,
                                  self.mean_img, self.predict)
            di.daemon = True
            global miniq_flag
            if miniq_flag:
                miniq.put(di)
            else:
                di.start()
                miniq_flag = True
        logger.debug('\tMid Minibatch %d %d %d', batch_idx, self.num_iter_gpu,
                     self.gpu_onque)

        for i in range(self.num_iter_gpu):
            self.gpu_onque = (self.gpu_onque + 1) % self.minis_per_macro
            gt = GPUTransfer(self.gpu_onque, self.mini_batch_queue,
                             self.gpu_queue, self.backend, self.ring_buffer)
            gt.daemon = True
            global gpuq_flag
            if gpuq_flag:
                gpuq.put(gt)
            else:
                gt.start()
                gpuq_flag = True

        inputs_be, targets_be, labels_be = self.gpu_queue.get(block=True)
        self.gpu_queue.task_done()
        self.num_iter_mini = 1
        self.num_iter_gpu = 1

        return inputs_be, targets_be, labels_be

    def has_set(self, setname):
        return True if (setname in ['train', 'validation']) else False
