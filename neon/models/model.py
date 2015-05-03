# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic Model interface.  Defines the operations and parameters any model
must support.
"""

import logging

logger = logging.getLogger(__name__)


class Model(object):
    """
    Abstract base model class.  Identifies operations to be implemented.
    """

    def fit(self, datasets):
        """
        Utilize the passed dataset(s) to train a model (learn model
        parameters).

        :param datasets: collection of datasets stored in an approporiate
                         backend.
        :type datasets: tuple of neon.datasets.Dataset objects
        """
        raise NotImplementedError()

    def predict(self, datasets):
        """
        Utilize a fit model to generate predictions against the datasets
        provided.

        :param datasets: collection of datasets stored in an approporiate
                         backend.
        :type datasets: tuple of neon.datasets.Dataset objects
        """
        raise NotImplementedError()

    def get_params(self):
        np_params = dict()
        for i, ll in enumerate(self.layers):
            if ll.has_params:
                lkey = ll.name + '_' + str(i)
                np_params[lkey] = ll.get_params()
        np_params['epochs_complete'] = self.epochs_complete
        return np_params

    def set_params(self, params_dict):
        for i, ll in enumerate(self.layers):
            if ll.has_params:
                lkey = ll.name + '_' + str(i)
                ll.set_params(params_dict[lkey])
        self.epochs_complete = params_dict['epochs_complete']
