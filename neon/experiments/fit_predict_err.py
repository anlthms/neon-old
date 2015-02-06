# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned), then performance
is evaluated on the predictions made.
"""

import logging

from neon.experiments.fit import FitExperiment

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
        self.report_sets = []
        self.metrics = []
        super(FitPredictErrorExperiment, self).__init__(**kwargs)

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        # Load the data and train the model.
        super(FitPredictErrorExperiment, self).run()
        self.model.predict_and_report(self.dataset)

        # Report error metrics.
        for setname in self.report_sets:
            res = self.model.predict_fullset(self.dataset, setname)
            for metric in self.metrics:
                val = self.model.report(*res, metric=metric)
                logger.info('%s set %s %.5f', setname, metric, val)
