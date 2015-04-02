# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a saved model is loaded and predictions made on
live samples of data. 
"""

import logging
import os

from neon.experiments.fit import FitExperiment

logger = logging.getLogger(__name__)


class PredictLiveExperiment(FitExperiment):
    """
    A pre-fit model is loaded and inference performed in real-time. 
    """

    def __init__(self, **kwargs):
        super(PredictLiveExperiment, self).__init__(**kwargs)

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        logger.info('Ready to perform inference')
        self.dataset.set_batch_size(self.model.batch_size)
        self.dataset.backend = self.backend
        self.dataset.load()
        self.model.predict_live_init(self.dataset)
        while True:
            try:
                result = self.model.predict_live()
                logger.info(result)
            except KeyboardInterrupt:
                self.dataset.close()
                return
