"""
ImageNet 1k dataset
More information at: http://www.image-net.org/download-imageurls
Sign up for an ImageNet account to download the dataset!
"""

import logging
import numpy as np
import os
import tarfile
import cPickle
#from make-data.pyext._MakeDataPyExt import resizeJPEG
#from scipy import misc
import Image
from StringIO import StringIO
import scipy.io
from random import shuffle
from time import time
from neon.util.compat import CUDA_GPU
if CUDA_GPU:
    import neon.backends.gpu

from neon.datasets.dataset import Dataset
from neon.util.compat import MPI_INSTALLED
from neon.util.persist import deserialize


logger = logging.getLogger(__name__)


class I1K(Dataset):

    """
    Sets up a ImageNet-1000 dataset.

    Attributes:
        url (str): where to find the source data
        backend (neon.backends.Backend): backend used for this data
        inputs (dict): structure housing the loaded train/test/validation
                       input data
        targets (dict): structure housing the loaded train/test/validation
                        target data

    Kwargs:
        repo_path (str, optional): where to locally host this dataset on disk

    """
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self, **kwargs):
        self.dist_flag = False
        self.macro_batched = False
        self.start_train_batch = -1
        self.end_train_batch = -1
        self.start_val_batch = -1
        self.end_val_batch = -1
        self.__dict__.update(kwargs)
        if self.dist_flag:
            if MPI_INSTALLED:
                from mpi4py import MPI
                self.comm = MPI.COMM_WORLD
                # for now require that comm.size is a square and divides 32
                if self.comm.size not in [1, 4, 16]:
                    raise AttributeError('MPI.COMM_WORLD.size not compatible')
            else:
                raise AttributeError("dist_flag set but mpi4py not installed")
        #if self.macro_batched:
        #    self.macro_batch_size = 3072


    # def fetch_dataset(self, save_dir):
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     repo_gz_file = os.path.join(save_dir, os.path.basename(self.url))
    #     if not os.path.exists(repo_gz_file):
    #         self.download_to_repo(self.url, save_dir)

    #     data_file = os.path.join(save_dir, 'cifar-10-batches-py', 'test_batch')
    #     if not os.path.exists(data_file):
    #         logger.info('untarring: %s' % repo_gz_file)
    #         infile = tarfile.open(repo_gz_file)
    #         infile.extractall(save_dir)
    #         infile.close()

    # def sample_training_data(self):
    #     if self.sample_pct != 100:
    #         train_idcs = np.arange(self.inputs['train'].shape[0])
    #         ntrain_actual = (self.inputs['train'].shape[0] *
    #                          int(self.sample_pct) / 100)
    #         np.random.shuffle(train_idcs)
    #         train_idcs = train_idcs[0:ntrain_actual]
    #         self.inputs['train'] = self.inputs['train'][train_idcs]
    #         self.targets['train'] = self.targets['train'][train_idcs]

    # def adjust_for_dist(self):
    #     # computes the indices to load from input data for the dist case

    #     comm_rank = self.comm.rank
    #     self.dist_indices = []
    #     img_width = 32
    #     img_2d_size = img_width ** 2
    #     img_size = img_2d_size * 3

    #     if self.dist_mode == 'halopar':
    #         # todo: will change for different x/y dims for comm_per_dim
    #         self.comm_per_dim = int(np.sqrt(self.comm.size))
    #         px_per_dim = img_width / self.comm_per_dim
    #         r_i = []
    #         c_i = []
    #         # top left corner in 2-D image
    #         for row in range(self.comm_per_dim):
    #             for col in range(self.comm_per_dim):
    #                 r_i.append(row * px_per_dim)
    #                 c_i.append(col * px_per_dim)
    #         for ch in range(3):
    #             for r in range(r_i[comm_rank], r_i[comm_rank] + px_per_dim):
    #                 self.dist_indices.extend(
    #                     [ch * img_2d_size + r * img_width + x for x in range(
    #                         c_i[comm_rank], c_i[comm_rank] + px_per_dim)])
    #     elif self.dist_mode == 'vecpar':
    #         start_idx = 0
    #         for j in range(comm_rank):
    #             start_idx += (img_size // self.comm.size +
    #                           (img_size % self.comm.size > j))
    #         nin = (img_size // self.comm.size +
    #                (img_size % self.comm.size > comm_rank))
    #         self.dist_indices.extend(range(start_idx, start_idx + nin))
    #     elif self.dist_mode == 'datapar':
    #         raise NotImplementedError('support for datapar not implemented')

    # def load_file(self, filename, nclasses):
    #     logger.info('loading: %s' % filename)
    #     dict = deserialize(filename)

    #     full_image = np.float32(dict['data'])
    #     full_image /= 255.

    #     if self.dist_flag:
    #         # read corresponding 'quad'rant of the image
    #         data = full_image[:, self.dist_indices]
    #     else:
    #         data = full_image

    #     labels = np.array(dict['labels'])
    #     onehot = np.zeros((len(labels), nclasses), dtype=np.float32)
    #     for col in range(nclasses):
    #         onehot[:, col] = (labels == col)
    #     return (data, onehot)

    def load(self):
        if self.inputs['train'] is not None:
            return
        if 'repo_path' in self.__dict__:
            if self.dist_flag:
                self.adjust_for_dist()
                ncols = len(self.dist_indices)
            else:
                ncols = 256 * 256 * 3

            self.nclasses = 1000
            load_dir = os.path.join(self.load_path,
                                    self.__class__.__name__)
            save_dir = os.path.join(self.repo_path,
                                    self.__class__.__name__)
            self.save_dir = save_dir
            # for now assuming that dataset is already there
            # ToS of imagenet prohibit distribution of URLs
            # self.fetch_dataset(save_dir)

            #from krizhevsky's make-data.py
            ILSVRC_TRAIN_TAR = os.path.join(load_dir, 'ILSVRC2012_img_train.tar')
            ILSVRC_VALIDATION_TAR = os.path.join(load_dir, 'ILSVRC2012_img_val.tar')
            ILSVRC_DEVKIT_TAR = os.path.join(load_dir, 'ILSVRC2012_devkit_t12.tar.gz')
            labels_dic, label_names, validation_labels = self.parse_devkit_meta(ILSVRC_DEVKIT_TAR)

            with self.open_tar(ILSVRC_TRAIN_TAR, 'training tar') as tf:
                synsets = tf.getmembers()
                synset_tars = [tarfile.open(fileobj=tf.extractfile(s)) for s in synsets]
                synset_tars=synset_tars[:4] #todo: delete this line
                logger.info("Loaded synset tars.")
                logger.info("Building training set image list (this can take 10-20 minutes)...")

                train_jpeg_files = []
                for i,st in enumerate(synset_tars):
                    if i % 100 == 0:
                        logger.info("%d%% ..." , int(round(100.0 * float(i) / len(synset_tars))))
                    train_jpeg_files += [st.extractfile(m) for m in st.getmembers()]
                    st.close()
                
                #self.pickle(train_file_path, train_jpeg_files)
                #print train_jpeg_files[0].read()[0:10]
                shuffle(train_jpeg_files)
                #print train_jpeg_files[0].read()[0:10]
                train_labels = [[labels_dic[jpeg.name[:9]]] for jpeg in train_jpeg_files]
                logger.info("created list of jpg files")
                
                self.CROP_TO_SQUARE          = True
                self.OUTPUT_IMAGE_SIZE       = 228
                # Number of threads to use for JPEG decompression and image resizing.
                self.NUM_WORKER_THREADS      = 8
                self.OUTPUT_BATCH_SIZE = 3072 #macro batch size
                self.max_file_index = 3072 #np.floor(self.sample_pct*len(train_jpeg_files))
                jpeg_file_sample = train_jpeg_files[0:self.max_file_index]
                label_sample = train_labels[0:self.max_file_index]
                
                self.val_max_file_index = 3072 #00
                val_label_sample = validation_labels[0:self.val_max_file_index]

                # todo 2: implement macro batching [will require changing model code]
                if self.macro_batched:
                    # Write training batches
                    self.num_train_macro_batches = self.write_batches(
                        os.path.join(save_dir, 'macro_batches'),
                                     'training', 0,
                                     label_sample, jpeg_file_sample)
                    with self.open_tar(ILSVRC_VALIDATION_TAR, 'validation tar') as tf:
                        validation_jpeg_files = sorted([tf.extractfile(m) for m in tf.getmembers()], key=lambda x:x.name)
                        val_file_sample = validation_jpeg_files[0:self.val_max_file_index]
                        self.num_val_macro_batches = self.write_batches(
                            os.path.join(save_dir, 'macro_batches'),
                            'validation', 0,
                            val_label_sample,
                            val_file_sample)
                    self.cur_train_macro_batch = 0
                    self.cur_train_mini_batch = 0
                    self.cur_val_macro_batch = 0
                    self.cur_val_mini_batch = 0
                else:
                    # one big batch (no macro-batches)
                    #todo 1: resize in a multithreaded manner
                    jpeg_mat = self.resizeJPEG([jpeg.read() for jpeg in jpeg_file_sample], as_string=False)
                    
                    self.inputs['train'] = jpeg_mat
                    # convert labels to one hot
                    tmp = np.zeros((self.nclasses, self.max_file_index), dtype='float32')
                    for col in range(self.nclasses):
                        tmp[col] = label_sample == col
                    self.targets['train'] = tmp
                    print 'done loading training data'

                    with self.open_tar(ILSVRC_VALIDATION_TAR, 'validation tar') as tf:
                        validation_jpeg_files = sorted([tf.extractfile(m) for m in tf.getmembers()], key=lambda x:x.name)
                        val_file_sample = validation_jpeg_files[0:self.val_max_file_index]
                        jpeg_mat = self.resizeJPEG([jpeg.read() for jpeg in val_file_sample], as_string=False)

                        self.inputs['test'] = jpeg_mat
                        tmp = np.zeros((self.nclasses, self.max_file_index), dtype='float32')
                        for col in range(self.nclasses):
                            tmp[col] = val_label_sample == col
                        self.targets['test'] = tmp
            
                    print "done loading imagenet data"
                    print self.inputs['train'].shape
                    print self.inputs['test'].shape
                    self.format()
        else:
            raise AttributeError('repo_path not specified in config')

    def resizeJPEG(self, jpeg_strings, as_string=False):
        if as_string:
            tgt = []
        else:
            # as numpy array, row order
            tgt = np.empty((len(jpeg_strings), (self.OUTPUT_IMAGE_SIZE**2) *3), dtype='float32')
        for i, jpeg_string in enumerate(jpeg_strings):
            #print jpeg_string[0:10]
            img = Image.open(StringIO(jpeg_string))
            # resize
            min_dim = np.min(img.size)
            scale_factor = np.float(self.OUTPUT_IMAGE_SIZE) / min_dim
            #print img.size, scale_factor
            new_w = np.int(np.rint(scale_factor* img.size[0]))
            new_h = np.int(np.rint(scale_factor* img.size[1]))
            img = img.resize((new_w, new_h)) #todo: interp mode?
            #print new_w, new_h
            # crop
            if self.CROP_TO_SQUARE:
                crop_start_x = (new_w - self.OUTPUT_IMAGE_SIZE) / 2;
                crop_start_y = (new_h - self.OUTPUT_IMAGE_SIZE) / 2;
                #print crop_start_x, crop_start_y, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE
                img = img.crop((crop_start_x, crop_start_y,
                               crop_start_x+self.OUTPUT_IMAGE_SIZE,
                               crop_start_y+self.OUTPUT_IMAGE_SIZE))
                #print img.size, img.mode

            else:
                raise NotImplementedError
            if as_string:
                f = StringIO()
                img.save(f, "JPEG")
                tgt.append(f.getvalue())
            else:
                # this is still in row order                
                if img.mode == 'L': #greyscale
                    logger.debug('greyscale image found... tiling')
                    tgt[i] = np.tile(np.array(img, dtype='float32').reshape((1,-1)),3)
                else:
                    tgt[i] = np.array(img, dtype='float32').reshape((1,-1))
            
        return tgt
    
    def get_macro_batch(self, batch_type, macro_batch_index, raw_targets=False):

        batch_path = os.path.join(self.save_dir, 'macro_batches', '%s_batch_%d' % (batch_type, macro_batch_index))
        j=0
        self.jpeg_strings = self.unpickle(os.path.join(batch_path, '%s_batch_%d.%d' % (batch_type, macro_batch_index, j/self.OUTPUT_BATCH_SIZE)))

        # during run time extract labels
        labels = self.jpeg_strings['labels']
        
        if not raw_targets:
            self.targets_macro = np.zeros((self.nclasses, self.OUTPUT_BATCH_SIZE), dtype='float32')
            for col in range(self.nclasses):
                self.targets_macro[col] = labels == col
        else:
            self.targets_macro = np.asarray(labels).reshape((1,-1))

    def get_mini_batch(self, batch_size, batch_type, raw_targets=False):
        if self.OUTPUT_BATCH_SIZE % batch_size != 0:
            raise ValueError('self.OUTPUT_BATCH_SIZE % batch_size != 0')
        else:
            num_minibatches_in_macro = self.OUTPUT_BATCH_SIZE / batch_size
            #print self.OUTPUT_BATCH_SIZE,batch_size, 'num_minibatches_in_macro=', num_minibatches_in_macro

        #keep track of most recent batch and return the next batch
        if batch_type == 'training':
            cur_mini_batch_id = self.cur_train_mini_batch
            if cur_mini_batch_id == 0:
                #when cur_mini_batch is 0 load macro batch
                print "train processing macro batch", self.cur_train_macro_batch
                self.get_macro_batch(batch_type, self.cur_train_macro_batch, raw_targets)
                self.cur_train_macro_batch += 1
                if (self.cur_train_macro_batch >= self.num_train_macro_batches and
                    self.end_train_batch == -1):
                    self.cur_train_macro_batch = 0
                elif self.cur_train_macro_batch > self.end_train_batch:
                    self.cur_train_macro_batch = self.start_train_batch
        elif batch_type == 'validation':
            cur_mini_batch_id = self.cur_val_mini_batch
            if cur_mini_batch_id == 0:
                print "val processing macro batch", self.cur_val_macro_batch
                self.get_macro_batch(batch_type, self.cur_val_macro_batch, raw_targets)
                self.cur_val_macro_batch += 1
                if (self.cur_val_macro_batch >= self.num_val_macro_batches and
                    self.end_val_batch == -1):
                    self.cur_val_macro_batch = 0
                elif self.cur_val_macro_batch > self.end_val_batch:
                    self.cur_val_macro_batch = self.start_val_batch
        else:
            raise ValueError('Invalid batch_type in get_batch')

        #provide mini batch from macro batch
        inputs = np.empty(((self.OUTPUT_IMAGE_SIZE**2) *3, batch_size), dtype='float32')
        start_idx = cur_mini_batch_id*batch_size
        end_idx = (cur_mini_batch_id+1)* batch_size

        # convert jpeg string to numpy array
        for i, jpeg_string in enumerate(self.jpeg_strings['data'][start_idx:end_idx]):
            img = Image.open(StringIO(jpeg_string))
            if img.mode == 'L': #greyscale
                logger.debug('greyscale image found... tiling')
                inputs[:,i,np.newaxis] = np.tile(np.array(img, dtype='float32').reshape((-1,1)),(3,1))
            else:
                inputs[:,i,np.newaxis] = np.array(img, dtype='float32').reshape((-1,1))

        targets = self.targets_macro[:,start_idx:end_idx]

        # serialize

        #todo: threaded conversion of jpeg strings to numpy array
        # todo: threaded load of next batch while compute of current batch
        #todo:
        #test file reading speeds 3M, 30M, 90M, 300M
        #multithreaded reads
        #multiple MPI processes reading diff files from same disk
        
        if batch_type == 'training':
            self.cur_train_mini_batch += 1
            if self.cur_train_mini_batch >= num_minibatches_in_macro:
                self.cur_train_mini_batch = 0
        elif batch_type == 'validation':
            self.cur_val_mini_batch += 1
            if self.cur_val_mini_batch >= num_minibatches_in_macro:
                self.cur_val_mini_batch = 0

        # TODO: resize image to 224x224 here224

        #if CUDA_GPU and type(self.backend) == neon.backends.gpu.GPU:
        #    return self.backend.array(inputs), self.backend.array(targets)
        #else:
        return self.backend.array(inputs), self.backend.array(targets)

    # def __getstate__(self):
    #     self._blacklist = ['start_train_batch', 'end_train_batch', 'start_val_batch', 'end_val_batch']
    #     return {k: v for k, v in self.__dict__.iteritems() if k not in self._blacklist}
    
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    # code below from Alex Krizhevsky's cuda-convnet2 library, make-data.py
    # Copyright 2014 Google Inc. All rights reserved.
    # 
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    # 
    #    http://www.apache.org/licenses/LICENSE-2.0
    # 
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    ##########################################################################
    def open_tar(self, path, name):
        if not os.path.exists(path):
            print "ILSVRC 2012 %s not found at %s. Make sure to set ILSVRC_SRC_DIR correctly at the top of this file (%s)." % (name, path, sys.argv[0])
            sys.exit(1)
        return tarfile.open(path)

    def pickle(self, filename, data):
        with open(filename, "w") as fo:
            cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

    def unpickle(self, filename):
        fo = open(filename, 'r')
        contents = cPickle.load(fo)
        fo.close()
        return contents
    
    def makedir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def parse_devkit_meta(self, ILSVRC_DEVKIT_TAR):
        tf = self.open_tar(ILSVRC_DEVKIT_TAR, 'devkit tar')
        fmeta = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/meta.mat'))
        meta_mat = scipy.io.loadmat(StringIO(fmeta.read()))
        labels_dic = dict((m[0][1][0], m[0][0][0][0]-1) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
        label_names_dic = dict((m[0][1][0], m[0][2][0]) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
        label_names = [tup[1] for tup in sorted([(v,label_names_dic[k]) for k,v in labels_dic.items()], key=lambda x:x[0])]

        fval_ground_truth = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
        validation_ground_truth = [[int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
        tf.close()
        return labels_dic, label_names, validation_ground_truth

    # following functions are for creating macrobatches
    def partition_list(self, l, partition_size):
        divup = lambda a,b: (a + b - 1) / b
        return [l[i*partition_size:(i+1)*partition_size] for i in xrange(divup(len(l),partition_size))]

    def write_batches(self, target_dir, name, start_batch_num, labels, jpeg_files):
        jpeg_files = self.partition_list(jpeg_files, self.OUTPUT_BATCH_SIZE)
        labels = self.partition_list(labels, self.OUTPUT_BATCH_SIZE)
        self.makedir(target_dir)
        logger.debug("Writing %s batches..." % name)
        for i,(labels_batch, jpeg_file_batch) in enumerate(zip(labels, jpeg_files)):
            t = time()
            jpeg_strings = self.resizeJPEG([jpeg.read() for jpeg in jpeg_file_batch],
                                           as_string=True)
            batch_path = os.path.join(target_dir, '%s_batch_%d' % (name, start_batch_num + i))
            self.makedir(batch_path)
            # no subbatch support for : do we really need them?
            #for j in xrange(0, len(labels_batch), self.OUTPUT_SUB_BATCH_SIZE):
            j = 0
            self.pickle(os.path.join(batch_path, '%s_batch_%d.%d' % (name, start_batch_num + i, j/self.OUTPUT_BATCH_SIZE)), 
                   {'data': jpeg_strings[j:j+self.OUTPUT_BATCH_SIZE],
                    'labels': labels_batch[j:j+self.OUTPUT_BATCH_SIZE]})
            logger.debug("Wrote %s (%s batch %d of %d) (%.2f sec)" % (batch_path, name, i+1, len(jpeg_files), time() - t))
        return i + 1


