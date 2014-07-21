"""
Defines how to initialize and run an experiment.
"""

import cPickle
import logging
import os

from util import Factory

logger = logging.getLogger(__name__)


def deserialize(load_path):
        logger.info("deserializing object from: %s" % load_path)
        return cPickle.load(open(load_path))


def serialize(obj, save_path):
        logger.info("serializing %s to: %s" % (str(obj), save_path))
        cPickle.dump(obj, open(save_path, 'wb'), -1)


class Experiment(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def run(self):
        backend = Factory.create(**self.backend)
        datasets = []
        for dataset in self.datasets:
            dataset['backend'] = backend
            if 'pkl_path' in dataset:
                dpath = dataset['pkl_path']
                if os.path.exists(dpath):
                    ds = deserialize(dpath)
                else:
                    ds = Factory.create(**dataset)
                    # TODO: fix serialization.  Can't serialize backend module
                    # serialize(ds, dataset['pkl_path'])
            else:
                ds = Factory.create(**dataset)
            datasets.append(ds)
        model = Factory.create(**self.model)
        model.fit(datasets)
        predictions = model.predict(datasets)
        model.error_metrics(datasets, predictions)
