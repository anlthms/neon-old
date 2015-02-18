import argparse as argp
import cPickle
import numpy as np
import os
import pandas as pd
import sys
import yaml
from multiprocessing import Pool
from PIL import Image
from StringIO import StringIO
from time import time

def pickle(filename, data):
    with open(filename, "w") as fo:
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

# NOTE: We have to leave this helper function out of the class and use the 
#       global variable hack so that we can use multiprocess pool.map
def proc_img(imgfile):
    tgt_size = TARGET_SIZE
    square_crop = SQUARE_CROP
    im = Image.open(imgfile)

    # This part does the processing
    scale_factor = tgt_size / np.float32(min(im.size))
    (wnew, hnew) = map(lambda x: int(round(scale_factor * x)), im.size)

    if scale_factor != 1:
        filt = Image.BICUBIC if scale_factor > 1 else Image.ANTIALIAS
        im = im.resize((wnew, hnew), filt)

    if square_crop is True:
        (cx, cy) = map(lambda x: (x - tgt_size) / 2, (wnew, hnew))
        im = im.crop((cx, cy, tgt_size, tgt_size))

    buf = StringIO()
    im.save(buf, format= 'JPEG')
    return buf.getvalue()

class BatchWriter(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        global TARGET_SIZE, SQUARE_CROP
        TARGET_SIZE = self.tgt_size
        SQUARE_CROP = self.square_crop

    def __str__(self):
        pairs = map(lambda a: a[0] + ': ' + a[1],
                     zip(self.__dict__.keys(),
                         map(str, self.__dict__.values())))
        return "\n".join(pairs)

    def parse_file_list(self, infile, doshuffle):
        compression = 'gzip' if infile.endswith('.gz') else None
        df = pd.read_csv(infile, compression=compression)
        if doshuffle:
            df.reindex(np.random.permutation(df.index))

        lk = filter(lambda x: x.startswith('l'), df.keys())
        tk = filter(lambda x: x.startswith('t'), df.keys())

        labels = {ll: np.array(df[ll].values, np.int32) for ll in lk}
        targets = np.array(df[tk].values, np.float32) if len(tk) > 0 else None
        imfiles = df['filename'].values

        return imfiles, labels, targets

    def write_batches(self, name, start_num, labels, imfiles, targets=None):
        psz = self.batch_size
        nparts = (len(imfiles) + psz - 1) / psz

        imfiles = [imfiles[i*psz:(i+1)*psz] for i in xrange(nparts)]

        if targets is not None:
            targets = [targets[i*psz:(i+1)*psz] for i in xrange(nparts)]

        labels = [{k:v[i*psz:(i+1)*psz] for k,v in labels.iteritems()}
                  for i in xrange(nparts)]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print "Writing %s batches..." % name
        for i, jpeg_file_batch in enumerate(imfiles):
            t = time()
            pool = Pool(processes=self.num_workers)
            jpeg_strings = pool.map(proc_img, jpeg_file_batch)
            labels_batch = labels[i]
            targets_batch = None if targets is None else targets[i]
            batchfile = os.path.join(self.output_dir,
                                     'data_batch_%d' % (start_num + i))
            pickle(batchfile,
                   {'data': jpeg_strings,
                    'labels': labels_batch,
                    'targets': targets_batch})
            print "Wrote to %s (%s batch %d of %d) (%.2f sec)" % (
                self.output_dir, name, i + 1, len(imfiles), time() - t)
        return i + 1

    def run(self):
        idx = 0
        for setname, t_or_v in zip(['train', 'validation', 'test'],
                     [self.train_file, self.validation_file, self.test_file]):
            if t_or_v is not None and os.path.exists(t_or_v):
                doshuffle = True if setname == 'train' else False
                imfiles, labels, targets = self.parse_file_list(t_or_v, doshuffle)
                idx = self.write_batches(setname, idx, labels, imfiles, targets)
            else:
                print 'Skipping {}, file missing'.format(setname)

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
