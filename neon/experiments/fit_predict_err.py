# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned), then performance
is evaluated on the predictions made.
"""

import logging
import os

from neon.util.persist import serialize
from neon.experiments.fit import FitExperiment
from neon.util.param import opt_param

logger = logging.getLogger(__name__)


class FitPredictErrorExperiment(FitExperiment):
    """
    In this `Experiment`, a model is first trained on a training dataset to
    learn a set of parameters, then these parameters are used to generate
    predictions on specified test datasets, and the resulting performance is
    measured then returned.

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
        super(FitPredictErrorExperiment, self).__init__(**kwargs)
        opt_param(self, ['inference_sets'], [])
        opt_param(self, ['inference_metrics'], [])
        if len(self.inference_metrics) != 0 and len(self.inference_sets) == 0:
            raise AttributeError('inference_metrics specified without '
                                 'inference_sets')

    def save_results(self, dataset, setname, data, dataname):
        filename = os.path.join(dataset.repo_path, dataset.__class__.__name__,
                                '{}-{}.pkl'.format(setname, dataname))
        serialize(data.asnumpyarray().T, filename)

    def run(self):
        """
        Actually carry out each of the experiment steps.

        Returns:
            dict: of inference_metric names, each entry of which is a dict
                  containing inference_set name keys, and actual metric values
        """
        result = dict()
        # Load the data and train the model.
        super(FitPredictErrorExperiment, self).run()
        # TODO: cleanup the call below to remove duplication with other
        # reporting.
        self.model.predict_and_report(self.dataset)

        # Report error metrics.
        for setname in self.inference_sets:
            if not self.dataset.has_set(setname):
                continue
            outputs, targets = self.model.predict_fullset(self.dataset,
                                                          setname)
            self.save_results(self.dataset, setname, outputs, 'inference')
            self.save_results(self.dataset, setname, targets, 'targets')
            for metric in self.inference_metrics:
                val = self.model.report(targets, outputs, metric=metric)
                logger.info('%s set %s %.5f', setname, metric, val)
                if metric not in result:
                    result[metric] = dict()
                result[metric][setname] = val
        return result
