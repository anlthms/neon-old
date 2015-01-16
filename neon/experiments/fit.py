# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned)
"""

import logging

from neon.experiments.experiment import Experiment
from neon.util.compat import MPI_INSTALLED
from neon.util.persist import serialize

logger = logging.getLogger(__name__)


class FitExperiment(Experiment):

    """
    In this `Experiment`, a model is trained on a training dataset to
    learn a set of parameters

    Note that a pre-fit model may be loaded depending on serialization
    parameters (rather than learning from scratch).  The same may also apply to
    the datasets specified.

    Kwargs:
        backend (neon.backends.Backend): The backend to associate with the
                                            datasets to use in this experiment
    TODO:
        add other params
    """

    def __init__(self, **kwargs):
        # default dist_flag to False
        self.dist_flag = False
        self.__dict__.update(kwargs)
        for req_param in ['backend', 'dataset', 'model']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """
        # load the dataset, save it to disk if specified
        self.dataset.set_batch_size(self.model.batch_size)
        if not hasattr(self.dataset, 'backend'):
            self.dataset.backend = self.backend
        self.dataset.load()
        if hasattr(self.dataset, 'serialized_path'):
            serialize(self.dataset, self.dataset.serialized_path)

        # fit the model to the data, save it if specified
        if not hasattr(self.model, 'backend'):
            self.model.backend = self.backend
        if not hasattr(self.model, 'fit_complete'):
            self.model.fit(self.dataset)
            self.model.fit_complete = True
        if hasattr(self.model, 'serialized_path'):
            if self.dist_flag:
                if MPI_INSTALLED:
                    from mpi4py import MPI
                    # serialize the model only at the root node
                    if MPI.COMM_WORLD.rank == 0:
                        serialize(self.model, self.model.serialized_path)
                else:
                    raise AttributeError("dist_flag set but mpi4py not "
                                         "installed")
            else:
                serialize(self.model, self.model.serialized_path)
