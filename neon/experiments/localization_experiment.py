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

    def run(self):
        """
        Calls into the existing model for localization
        """

        # load the data and train the model
        self.model.predict_and_localize(self.dataset)
