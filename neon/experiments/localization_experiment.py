# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a trained model is used to compute localization confidence
maps on large input images.
"""

import logging

from neon.experiments.fit import FitExperiment

logger = logging.getLogger(__name__)


class LocalizationExperiment(FitExperiment):
    """
    [TODO] This is not a real experiment yet.
    """
    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        # load the data and train the model
        #super(LocalizationExperiment, self).run() # this might set up a lot!
        self.model.predict_and_localize(self.dataset)
