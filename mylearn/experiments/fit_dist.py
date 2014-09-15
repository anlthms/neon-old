"""
Experiment in which a model is trained (parameters learned)
"""

import logging
import os

from mylearn.experiments.experiment import Experiment
from mylearn.util.persist import serialize, deserialize
from mpi4py import MPI

logger = logging.getLogger(__name__)


class FitExperiment(Experiment):
    """
    In this `Experiment`, a model is trained on a training dataset to
    learn a set of parameters

    Note that a pre-fit model may be loaded depending on serialization
    parameters (rather than learning from scratch).  The same may also apply to
    the datasets specified.

    Kwargs:
        backend (mylearn.backends.Backend): The backend to associate with the
                                            datasets to use in this experiment
        TODO: add other params
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for req_param in ['backend', 'datasets', 'model']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        # load and/or deserialize any unloaded datasets
        for ds_idx in range(len(self.datasets)):
            ds = self.datasets[ds_idx]
            if not hasattr(ds, 'backend'):
                ds.backend = self.backend
            if hasattr(ds, 'serialized_path'):
                if os.path.exists(ds.serialized_path):
                    self.datasets[ds_idx] = deserialize(ds.serialized_path + str(MPI.COMM_WORLD.rank) + '.pkl')
                else:
                    ds.load()
                    serialize(ds, ds.serialized_path)
            else:
                ds.load()

        # load or fit the model to the data
        if not hasattr(self.model, 'backend'):
            self.model.backend = self.backend
        if hasattr(self.model, 'serialized_path'):
            mpath = self.model.serialized_path
            if os.path.exists(mpath):
                self.model = deserialize(mpath)
            else:
                self.model.fit(self.datasets)
                serialize(self.model, mpath)
        else:
            self.model.fit(self.datasets)
