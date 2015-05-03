# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned)
"""

import logging

from neon.experiments.experiment import Experiment
from neon.util.param import req_param, opt_param
from neon.util.persist import serialize, deserialize

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
        self.datapar = False
        self.modelpar = False
        self.initialized = False
        self.__dict__.update(kwargs)
        req_param(self, ['dataset', 'model'])
        opt_param(self, ['backend'])
        opt_param(self, ['live'], False)

    def initialize(self, backend):
        if self.initialized:
            return
        self.backend = backend
        self.model.link()
        self.backend.par.init_model(self.model, self.backend)
        self.model.initialize(backend)
        self.initialized = True

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        # load the dataset, save it to disk if specified
        self.dataset.set_batch_size(self.model.batch_size)
        self.dataset.backend = self.backend
        self.dataset.load()
        if hasattr(self.dataset, 'serialized_path'):
            logger.warning('Ability to serialize dataset has been deprecated.')

        # fit the model to the data, save it if specified
        if not hasattr(self.model, 'backend'):
            self.model.backend = self.backend
        if not hasattr(self.model, 'epochs_complete'):
            self.model.epochs_complete = 0
        if hasattr(self.model, 'depickle'):
            import os
            mfile = os.path.expandvars(os.path.expanduser(self.model.depickle))
            if os.access(mfile, os.R_OK):
                self.model.set_params(deserialize(mfile))
            else:
                logger.info('Unable to find saved model %s, starting over',
                            mfile)
        if self.model.epochs_complete >= self.model.num_epochs:
            return
        if self.live:
            return

        self.model.fit(self.dataset)

        if hasattr(self.model, 'pickle'):
            self.model.uninitialize()
            if self.backend.rank() == 0:
                serialize(self.model.get_params(), self.model.pickle)

        if hasattr(self.model, 'serialized_path'):
            ''' TODO: With the line below active, get

              File "/home/users/urs/code/neon/neon/layers/fully_connected.py",
              line 58, in bprop
                self.backend.update_fc(out=upm[u_idx], inputs=inputs,
            IndexError: list index out of range

            when deserializing a partially trained model'''
            self.model.uninitialize()
            if self.backend.rank() == 0:
                serialize(self.model, self.model.serialized_path)
