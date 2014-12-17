# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned), then performance
is evaluated on the predictions made.
"""

import logging

from neon.experiments.fit import FitExperiment
from neon.models.balance import Balance
from neon.util.persist import ensure_dirs_exist
import numpy as np
logger = logging.getLogger(__name__)


class GenOutputExperiment(FitExperiment):
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
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # load the data and train the model
        super(GenOutputExperiment, self).run()
        if not hasattr(self, 'batchnum'):
            self.batchnum = 1
        if not hasattr(self, 'zparam'):
            self.zparam = 0.0
        ds = self.datasets[0]
        inputs = ds.get_inputs(train=True, test=True, validation=True)['train']
        inputs_batch = ds.get_batch(inputs, self.batchnum)

        if isinstance(self.model, Balance):
            self.model.generate_output(inputs_batch, self.zparam)
            outputs_batch = self.model.get_reconstruction_output()
        else:
            self.model.fprop(inputs_batch)
            outputs_batch = self.model.layers[-1].output

        # now dump these values out into a few files
        numtest = 72
        iraw = inputs_batch.raw()[:, :numtest]
        oraw = outputs_batch.raw()[:, :numtest]

        fshape = (28, 28)
        oimg = np.zeros((28*12, 28*12))
        for x in xrange(6):
            for y in xrange(12):
                idx = 6*y + x
                oimg[y*28:(y+1)*28, x*56:(x+1)*56] = np.hstack(
                    (iraw[:, idx].reshape(fshape),
                     oraw[:, idx].reshape(fshape)))

        plt.imshow(oimg, interpolation='nearest', cmap='gray')
        plt.savefig(ensure_dirs_exist('output_example'))
