import logging

import argparse as argp
import gzip
import imgworker
import numpy as np
import os
import yaml
from glob import glob
from multiprocessing import Pool
from neon.util.compat import range, StringIO
from neon.util.param import opt_param
from neon.util.persist import serialize
from time import time


TARGET_SIZE = None
SQUARE_CROP = True

logger = logging.getLogger(__name__)


# NOTE: We have to leave this helper function out of the class and use the
#       global variable hack so that we can use multiprocess pool.map
def proc_img(imgfile):
    from PIL import Image
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
        self.batch_size = self.macro_size
        global TARGET_SIZE, SQUARE_CROP
        TARGET_SIZE = self.output_image_size
        SQUARE_CROP = self.square_crop
        opt_param(self, ['file_pattern'], '*.jpg')
        opt_param(self, ['validation_pct'], 0.2)
        opt_param(self, ['class_samples_max'])
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
        posfiles = glob(os.path.join(self.in_dir, '1', self.file_pattern))
        negfiles = glob(os.path.join(self.in_dir, '0', self.file_pattern))

        np.random.shuffle(posfiles)
        np.random.shuffle(negfiles)

        if self.class_samples_max is not None:
            posfiles = posfiles[:self.class_samples_max]
            negfiles = negfiles[:self.class_samples_max]

        poslines = [(filename, 1, 0, 1) for filename in posfiles]
        neglines = [(filename, 0, 1, 0) for filename in negfiles]

        v_idxp = int(self.validation_pct * len(poslines))
        v_idxn = int(self.validation_pct * len(neglines))

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

        serialize({'ntrain': self.ntrain,
                   'nval': self.nval,
                   'train_start': self.train_start,
                   'val_start': self.val_start,
                   'macro_size': self.batch_size,
                   'train_mean': self.train_mean,
                   'val_mean': self.val_mean}, self.stats)

    def parse_file_list(self, infile):
        import pandas as pd
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
        batch_mean = np.zeros(accum_buf.shape, dtype=np.uint8)
        logger.info("Writing %s batches...", name)
        for i, jpeg_file_batch in enumerate(imfiles):
            t = time()
            pool = Pool(processes=self.num_workers)
            jpeg_strings = pool.map(proc_img, jpeg_file_batch)
            pool.close()
            targets_batch = None if targets is None else targets[i]
            labels_batch = labels[i]
            bfile = os.path.join(self.out_dir, 'data_batch_%d' % (start + i))
            serialize({'data': jpeg_strings,
                       'labels': labels_batch,
                       'targets': targets_batch},
                      bfile)
            logger.info("Wrote to %s (%s batch %d of %d) (%.2f sec)",
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
            logger.info("%s %s %s", sname, fname, start)
            if fname is not None and os.path.exists(fname):
                imgs, labels, targets = self.parse_file_list(fname)
                self.write_batches(sname, start, labels, imgs, targets)
            else:
                logger.info('Skipping %s, file missing', sname)


class BatchWriterDepth(BatchWriter):

    def write_batches(self, name, start, labels, imfiles, targets=None):
        psz = self.batch_size
        npts = (len(imfiles) + psz - 1) / psz

        imfiles = [imfiles[i*psz: (i+1)*psz] for i in range(npts)]

        if targets is not None:
            targets = [targets[i*psz: (i+1)*psz].T.copy() for i in range(npts)]

        labels = [{k: v[i*psz: (i+1)*psz] for k, v in labels.iteritems()}
                  for i in range(npts)]

        logger.info("Writing %s batches...", name)
        for i, jpeg_file_batch in enumerate(imfiles):
            t = time()
            pool = Pool(processes=self.num_workers)
            jpeg_strings = pool.map(proc_img, jpeg_file_batch)
            dpth_file_batch = map(lambda x: x.replace('left', 'depth'),
                                  jpeg_file_batch)
            dpth_strings = pool.map(proc_img, dpth_file_batch)
            pool.close()
            targets_batch = None if targets is None else targets[i]
            labels_batch = labels[i]
            bfile = os.path.join(self.out_dir, 'data_batch_%d' % (start + i))
            serialize({'data': jpeg_strings,
                       'dpth': dpth_strings,
                       'labels': labels_batch,
                       'targets': targets_batch},
                      bfile)
            logger.info("Wrote to %s (%s batch %d of %d) (%.2f sec)",
                        self.out_dir, name, i + 1, len(imfiles), time() - t)

    def run(self):
        self.write_csv_files()
        namelist = ['train', 'validation']
        filelist = [self.train_file, self.val_file]
        startlist = [self.train_start, self.val_start]
        for sname, fname, start in zip(namelist, filelist, startlist):
            logger.info("%s %s %s", sname, fname, start)
            if fname is not None and os.path.exists(fname):
                imgs, labels, targets = self.parse_file_list(fname)
                self.write_batches(sname, start, labels, imgs, targets)
            else:
                logger.info('Skipping %s, file missing', sname)


if __name__ == "__main__":
    parser = argp.ArgumentParser()
    parser.add_argument('--config', help='Configuration File', required=True)
    parser.add_argument('--dataset', help='Dataset name', required=True)

    args = parser.parse_args()
    with open(args.config) as f:
        ycfg = yaml.load(f)[args.dataset]
    bw = BatchWriter(**ycfg)
    print bw
    bw.run()
