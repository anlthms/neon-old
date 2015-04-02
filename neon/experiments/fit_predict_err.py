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
        opt_param(self, ['diagnostics'], {'timing': False, 'ranges': False})
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
        """

        # if the experiment includes timing diagnostics, decorate backend
        if self.diagnostics['timing']:
            self.backend.flop_timing_init(self.diagnostics['decorate_fc'],
                                          self.diagnostics['decorate_conv'],
                                          self.diagnostics['decorate_ew'])
            self.model.timing_plots = True

        # if the experiment includes parameter statistics
        if self.diagnostics['ranges']:
            from neon.diagnostics import ranges_decorators
            rd = ranges_decorators.Decorators(backend=self.backend,
                                              silent=self.diagnostics[
                                                                'silent'])
            rd.decorate(function_list=self.diagnostics)

        # Load the data and train the model.
        super(FitPredictErrorExperiment, self).run()
        self.model.predict_and_report(self.dataset)

        # visualization (if so requested)
        if self.diagnostics['timing']:
            from neon.diagnostics import timing_plots as tp
            tp.print_performance_stats(self.backend, logger)
        if self.diagnostics['ranges']:
            from neon.diagnostics import ranges_plots as rp
            rp.print_param_stats(self.backend, logger,
                                 self.diagnostics['filename'])

        # Report error metrics.
        for setname in self.inference_sets:
            outputs, targets = self.model.predict_fullset(self.dataset,
                                                          setname)
            self.save_results(self.dataset, setname, outputs, 'inference')
            self.save_results(self.dataset, setname, targets, 'targets')
            for metric in self.inference_metrics:
                val = self.model.report(targets, outputs, metric=metric)
                logger.info('%s set %s %.5f', setname, metric, val)
