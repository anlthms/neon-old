# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic Dataset interface.  Defines the operations any dataset should support.
"""

import logging
import os

from neon.backends.cpu import CPU
from neon.util.compat import PY3, CUDA_GPU

if PY3:
    import urllib.request as urllib
else:
    import urllib

logger = logging.getLogger(__name__)

if CUDA_GPU:
    logger.info("CUDA_GPU is set")
    import neon.backends.gpu


class Dataset(object):
    """
    Base dataset class. Defines interface operations.
    """

    backend = None
    inputs = {'train': None, 'test': None, 'validation': None}
    targets = {'train': None, 'test': None, 'validation': None}

    def __getstate__(self):
        """
        Defines what and how we go about serializing an instance of this class.
        In this case we also want to include and loaded datasets and backend
        references.

        Returns:
            dict: keyword args, plus current inputs, targets, backend
        """
        self.__dict__['backend'] = self.backend
        self.__dict__['inputs'] = self.inputs
        self.__dict__['targets'] = self.targets
        return self.__dict__

    def __setstate__(self, state):
        """
        Defines how we go about deserializing and loading an instance of this
        class from a specified state.

        Arguments:
            state (dict): attribute values to be loaded.
        """
        self.__dict__.update(state)
        if self.backend is None:
            # use CPU as a default backend
            self.backend = CPU()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def load(self, backend=None):
        """
        Makes the dataset data avilable for use.
        Needs to be implemented in every concrete Dataset child class.

        Arguments:
            backend (neon.backends.backend.Backend, optional): The
                     underlying data structure type used to hold this
                     data once loaded.  If None will use
                     `neon.backends.cpu.CPU`

        Raises:
            NotImplementedError: should be overridden in child class
        """
        raise NotImplementedError()

    def download_to_repo(self, url, repo_path):
        """
        Fetches the dataset to a local repository for future use.

        :param url: The external URI to a specific dataset
        :type url: str
        :param repo_path: The local path to write the fetched dataset to
        :type repo_path: str
        """
        logger.info("fetching: %s, saving to: %s", url, repo_path)
        urllib.urlretrieve(url, os.path.join(repo_path,
                                             os.path.basename(url)))

    def get_inputs(self, backend=None, train=True, test=False,
                   validation=False):
        """
        Loads and returns one or more input datasets.

        Arguments:
            backend (neon.backends.backend.Backend, None): The underlying
                    data structure type used to hold this data once loaded.
                    If None will use whatever is set for this class
            train (bool, optional): load a training target outcome dataset.
                                    Defaults to True.
            test (bool, optional): load a hold-out test target outcome dataset.
                                   Defaults to False.
            validation (bool, optional): Load a separate validation target
                                         outcome dataset.  Defaults to False.

        Returns:
            dict: of loaded datasets with keys train, test, validation
                  based on what was requested.  Each dataset is a
                  neon.backends.backend.Tensor instance.
        """
        res = dict()
        if self.inputs['train'] is None:
            if backend is not None:
                self.load(backend)
            else:
                self.load()
        if train and self.inputs['train'] is not None:
            res['train'] = self.inputs['train']
        if test and self.inputs['test'] is not None:
            res['test'] = self.inputs['test']
        if validation and self.inputs['validation'] is not None:
            res['validation'] = self.inputs['validation']
        return res

    def get_targets(self, backend=None, train=True, test=False,
                    validation=False):
        """
        Loads and returns one or more labelled outcome datasets.

        Arguments:
            backend (neon.backends.backend.Backend, None): The underlying
                    data structure type used to hold this data once loaded.
                    If None will use whatever is set for this class
            train (bool, optional): load a training target outcome dataset.
                                    Defaults to True.
            test (bool, optional): load a hold-out test target outcome dataset.
                                   Defaults to False.
            validation (bool, optional): Load a separate validation target
                                         outcome dataset.  Defaults to False.

        Returns:
            dict: of loaded datasets with keys train, test, validation
                  based on what was requested.  Each dataset is a
                  neon.backends.backend.Tensor instance.
        """
        # can't have targets without inputs, ensure these are loaded
        res = dict()
        if self.inputs['train'] is None:
            self.load()
        if train and self.inputs['train'] is not None:
            res['train'] = self.targets['train']
        if test and self.inputs['test'] is not None:
            res['test'] = self.targets['test']
        if validation and self.inputs['validation'] is not None:
            res['validation'] = self.targets['validation']
        return res

    def transpose_batches(self, data):
        """
        Transpose each minibatch within the dataset.
        """
        bs = self.batch_size
        nbatches = (data.shape[0] + bs - 1) / bs
        nrows = data.shape[1]
        batchwise = self.backend.zeros((nbatches * nrows, bs))
        for batch in xrange(nbatches):
            batchdata = data[batch * bs:(batch + 1) * bs].transpose()
            if CUDA_GPU and type(self.backend) == neon.backends.gpu.GPU:
                batchdata = batchdata.copy()
            ncols = batchdata.shape[1]
            assert ncols == bs
            batchwise[batch * nrows:(batch + 1) * nrows, 0:ncols] = (
                self.backend.array(batchdata))
        batchwise.nbatches = nbatches
        batchwise.nrows = nrows
        return batchwise

    def format(self):
        """
        Transforms the loaded data into the format expected by the
        backend. If a hardware accelerator device is being used,
        this function also copies the data to the device memory.
        """
        assert self.backend is not None
        for key in self.inputs:
            item = self.inputs[key]
            if item is not None:
                self.inputs[key] = self.transpose_batches(item)
        for key in self.targets:
            item = self.targets[key]
            if item is not None:
                self.targets[key] = self.transpose_batches(item)

    def get_batch(self, data, batch):
        return data[batch * data.nrows:(batch + 1) * data.nrows]
