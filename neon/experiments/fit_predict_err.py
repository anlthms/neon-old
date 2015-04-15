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

    def initialize(self, backend):
        if self.live:
            if not hasattr(self.dataset, 'live'):
                raise AttributeError('This dataset does not support '
                                     'live inference')
            self.model.batch_size = 1
            self.dataset.live = True
        super(FitPredictErrorExperiment, self).initialize(backend)

    def save_results(self, dataset, setname, data, dataname):
        out_dir = os.path.join(dataset.repo_path, dataset.__class__.__name__)
        if hasattr(dataset, 'save_dir'):
            out_dir = dataset.save_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        filename = os.path.join(out_dir, '{}-{}.pkl'.format(setname, dataname))
        serialize(data.asnumpyarray().T, filename)

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        # Load the data and train the model.
        super(FitPredictErrorExperiment, self).run()
        if self.live:
            self.predict_live()
            return

        self.model.predict_and_report(self.dataset)
        # Report error metrics.
        for setname in self.inference_sets:
            outputs, targets = self.model.predict_fullset(self.dataset,
                                                          setname)
            self.save_results(self.dataset, setname, outputs, 'inference')
            self.save_results(self.dataset, setname, targets, 'targets')
            for metric in self.inference_metrics:
                val = self.model.report(targets, outputs, metric=metric)
                logger.info('%s set %s %.5f', setname, metric, val)
        self.dataset.unload()

    def predict_live(self):
        self.model.predict_live_init(self.dataset)
        logger.info('Ready to perform live inference')
        while True:
            try:
                result = self.model.predict_live()
                logger.info(result)
            except KeyboardInterrupt:
                logger.info('Execution interrupted.')
                self.dataset.unload()
                break
