"""
Experiment in which a model is trained (parameters learned), then performance
is evaluated on the predictions made.
"""

import logging
import os

from mylearn.experiments.experiment import Experiment
from mylearn.util.factory import Factory
from mylearn.util.persist import serialize, deserialize

logger = logging.getLogger(__name__)


class FitPredictErrorExperiment(Experiment):
    """
    In this `Experiment`, a model is first trained on a training dataset to
    learn a set of parameters, then these parameters are used to generate
    predictions on specified test datasets, and the resulting performance is
    measured then returned.

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

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """
        backend = Factory.create(**self.backend)
        datasets = []
        for dataset in self.datasets:
            dataset['backend'] = backend
            if 'serialized_path' in dataset:
                dpath = dataset['serialized_path']
                if os.path.exists(dpath):
                    ds = deserialize(dpath)
                else:
                    ds = Factory.create(**dataset)
                    ds.load()
                    serialize(ds, dpath)
            else:
                ds = Factory.create(**dataset)
            datasets.append(ds)
        if 'serialized_path' in self.model:
            mpath = self.model['serialized_path']
            if os.path.exists(mpath):
                model = deserialize(mpath)
            else:
                model = Factory.create(**self.model)
                model.fit(datasets)
                serialize(model, mpath)
        else:
            model = Factory.create(**self.model)
            model.fit(datasets)
        if (self.model['type'] != 'mylearn.models.rbm.RBM') and (
                self.model['type'] != 'mylearn.models.dbn.DBN'):
            predictions = model.predict(datasets)
            model.error_metrics(datasets, predictions)
