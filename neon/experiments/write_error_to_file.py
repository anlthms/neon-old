# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned), then performance
is evaluated on the predictions made.
"""

import logging

from neon.experiments.fit import FitExperiment
from neon.util.compat import MPI_INSTALLED

logger = logging.getLogger(__name__)


class WriteErrorToFile(FitExperiment):
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
    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        # load the data and train the model
        super(WriteErrorToFile, self).run()

        prediction = self.model.predict_and_error(self.dataset)
        with open('neon_result_validation.txt', 'w') as f:
            f.write(str(prediction['validation'][0,0]))
