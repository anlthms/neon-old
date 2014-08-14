"""
Generic Dataset interface.  Defines the operations any dataset should support.
"""

import logging
import os

from mylearn.util.compat import PY3

if PY3:
    import urllib.request as urllib
else:
    import urllib


logger = logging.getLogger(__name__)


class Dataset(object):
    """
    Base dataset class. Defines interface operations.
    """

    backend = None

    def load(self, backend=None):
        """
        Makes the dataset data avilable for use.
        Needs to be implemented in every concrete Dataset child class.

        :param backend: The underlying data structure type used to hold this
                        data once loaded.  If None will use whatever is set
                        as default for this class
        :type backend: mylearn.backends.backend child class or None
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
        logger.info("fetching: %s, saving to: %s" % (url, repo_path))
        urllib.urlretrieve(url, os.path.join(repo_path,
                                             os.path.basename(url)))

    def get_inputs(self, backend=None, train=True, test=False,
                   validation=False):
        """
        Loads and returns one or more input datasets.

        Arguments:
            backend (mylearn.backends.backend.Backend, None): The underlying
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
                  mylearn.backends.backend.Tensor instance.
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
            backend (mylearn.backends.backend.Backend, None): The underlying
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
                  mylearn.backends.backend.Tensor instance.
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
