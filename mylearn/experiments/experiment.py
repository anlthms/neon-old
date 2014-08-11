"""
Defines how to initialize and run an experiment.
"""

import logging
import os
from ipdb import set_trace as trace
from mylearn.util.factory import Factory
from mylearn.util.persist import serialize, deserialize

logger = logging.getLogger(__name__)


class Experiment(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def run(self):
        print "(u) Backend:", self.backend # (u) what is this?
        trace()
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
        predictions = model.predict(datasets)
        model.error_metrics(datasets, predictions)
