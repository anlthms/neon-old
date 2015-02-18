# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned)
"""

import logging

from neon.experiments.experiment import Experiment
from neon.util.param import req_param, opt_param
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
        self.datapar = False
        self.modelpar = False
        self.__dict__.update(kwargs)
        req_param(self, ['dataset', 'model'])
        opt_param(self, ['backend'])

    def initialize(self, backend):
        self.backend = backend
        self.model.link()
        self.backend.par.init_model(self.model, self.backend)
        self.model.initialize(backend)

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        # load the dataset, save it to disk if specified
        if (not hasattr(self.dataset, 'dist_flag') or
                not self.dataset.dist_flag or (self.dataset.dist_mode !=
                                               'datapar')):
            self.dataset.set_batch_size(self.model.batch_size)
        self.dataset.backend = self.backend
        if hasattr(self.dataset, 'serialized_path'):
            serialize(self.dataset, self.dataset.serialized_path)

        # fit the model to the data, save it if specified
        if not hasattr(self.model, 'backend'):
            self.model.backend = self.backend
        if not hasattr(self.model, 'epochs_complete'):
            self.model.epochs_complete = 0
        if self.model.epochs_complete < self.model.num_epochs:
            self.model.fit(self.dataset)
        if hasattr(self.model, 'serialized_path'):
            if (hasattr(self.dataset, 'dist_flag') and self.dataset.dist_flag
                    and self.dataset.dist_mode == 'datapar'):
                if self.backend.mpi_rank == 0:
                    serialize(self.model, self.model.serialized_path)
            else:
                serialize(self.model, self.model.serialized_path)
