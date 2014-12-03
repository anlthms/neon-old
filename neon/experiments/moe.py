# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Class to generate spearmint runs for hyperparameter optimization.
TODO: Add spearmint installation as a new dependency:
        git clone https://github.com/Yelp/MOE.git
        pip install pyramid
        sudo port install boost

"""

import logging
import numpy as np

from neon.datasets.synthetic import UniformRandom
from neon.experiments.experiment import Experiment


logger = logging.getLogger(__name__)


class MOE(Experiment):
    """
    In this `Experiment`, a model is trained on a fake training dataset to
    validate the backprop code within the given model.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def save_state(self):

    def load_state(self):


    def check_layer(self, layer, inputs, targets):
        # Check up to this many weights.


    def run(self):
        """
        Actually carry out each of the experiment steps.
        """

