# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned)
"""

import logging
import os

from neon.experiments.experiment import Experiment
from neon.util.compat import MPI_INSTALLED, range
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
        # default dist_flag to False
        self.dist_flag = False
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
            ds.set_batch_size(self.model.batch_size)
            if not hasattr(ds, 'backend'):
                ds.backend = self.backend
            if hasattr(ds, 'serialized_path'):
                if self.dist_flag:
                    if MPI_INSTALLED:
                        from mpi4py import MPI
                        ds.serialized_path = ds.serialized_path.format(
                            rank=str(MPI.COMM_WORLD.rank),
                            size=str(MPI.COMM_WORLD.size))
                    else:
                        raise AttributeError("dist_flag set but mpi4py not "
                                             "installed")
                if os.path.exists(ds.serialized_path):
                    set_batches = False
                    if hasattr(self.datasets[ds_idx], 'start_train_batch'):
                        [tmp1, tmp2, tmp3, tmp4, tmp5] = [
                            self.datasets[ds_idx].start_train_batch,
                            self.datasets[ds_idx].end_train_batch,
                            self.datasets[ds_idx].start_val_batch,
                            self.datasets[ds_idx].end_val_batch,
                            self.datasets[ds_idx].num_processes]
                        set_batches = True
                    self.datasets[ds_idx] = deserialize(ds.serialized_path)
                    if set_batches:
                        [self.datasets[ds_idx].start_train_batch,
                         self.datasets[ds_idx].end_train_batch,
                         self.datasets[ds_idx].start_val_batch,
                         self.datasets[ds_idx].end_val_batch,
                         self.datasets[ds_idx].num_processes] = [
                            tmp1, tmp2, tmp3, tmp4, tmp5]
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
                if self.dist_flag:
                    if MPI_INSTALLED:
                        # deserialize the model at the root node only
                        # can change behavior depending on future use cases
                        if MPI.COMM_WORLD.rank == 0:
                            self.model = deserialize(mpath)
                    else:
                        raise AttributeError("dist_flag set but mpi4py not "
                                             "installed")
                self.model = deserialize(mpath)
            else:
                self.model.fit(self.datasets)
                if self.dist_flag:
                    if MPI_INSTALLED:
                        from mpi4py import MPI
                        # serialize the model only at the root node
                        if MPI.COMM_WORLD.rank == 0:
                            serialize(self.model, mpath)
                    else:
                        raise AttributeError("dist_flag set but mpi4py not "
                                             "installed")
                else:
                    serialize(self.model, mpath)
        else:
            self.model.fit(self.datasets)
