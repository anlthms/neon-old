# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment mainly for autoencoder type models where we want to visualize the
output when compared to the input.  Originally made for balance networks
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
    def __init__(self, **kwargs):
        super(GenOutputExperiment, self).__init__(**kwargs)
        if not hasattr(self, 'batchnum'):
            self.batchnum = 1
        if not hasattr(self, 'zparam'):
            self.zparam = 0.0
        for req_param in ['olayout', 'figure_filename', 'fshape']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # load the data and train the model
        super(GenOutputExperiment, self).run()

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
        reshape_dims = (self.olayout[1], self.olayout[0], 2,
                        self.fshape[0], self.fshape[1])
        figure_dims = (self.olayout[0] * self.fshape[0],
                       self.olayout[1] * self.fshape[1] * 2)
        numtest = self.olayout[0] * self.olayout[1]

        iraw = inputs_batch.raw()[:, :numtest]
        oraw = outputs_batch.raw()[:, :numtest]

        oimg = np.vstack([iraw, oraw]).transpose().reshape(
            reshape_dims).transpose(1, 3, 0, 2, 4).reshape(figure_dims)

        plt.imshow(oimg, interpolation='nearest', cmap='gray')
        plt.savefig(ensure_dirs_exist(self.figure_filename))
