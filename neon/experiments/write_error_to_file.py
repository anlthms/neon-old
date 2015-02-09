# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned), then performance
is evaluated on the predictions made.
"""

import logging
import os

from neon.experiments.fit import FitExperiment

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

    Attributes:
        backend (neon.backends.Backend): The backend to associate with the
                                         datasets to use in this experiment
        filename: name of the text file where the result should be stored
        item: one of 'test', 'train', 'validation', specifies which error
              should be written to file.
    """
    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

        # load the data and train the model
        super(WriteErrorToFile, self).run()
        if self.dataset.inputs[self.item] is not None:
            prediction = self.model.predict_and_error(self.dataset)
            with open(self.filename, 'w') as f:
                f.write(str(prediction[self.item]))
            logger.info("Writing '%s' error to %s" % (self.item,
                        os.getcwd() + '/' + self.filename))
        else:
            raise AttributeError("To perform WriteErrorToFile experiment "
                                 "please provide data '%s' set" % self.item)