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
        Utilize the passed datasets to train a model (learn model parameters).

        :param datasets:
        :type datasets: tuple of 
        """
        raise NotImplementedError()

    def predict(self, datasets):
        raise NotImplementedError()
