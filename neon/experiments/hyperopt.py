# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Simple random serach hyperparameter optimization.
[TODO]  Add spearmint installation as a new dependency:
        git clone https://github.com/Yelp/MOE.git
        pip install pyramid
        sudo port install boost
[TODO]  extract a "new set" of parameters from hyper-yaml
[TODO]  burn it to a normal hyper-yaml
[TODO]  run an experiment on the yaml
[TODO]  read the result of the result and add to list with parameters

"""

import logging
import numpy as np

from neon.datasets.synthetic import UniformRandom
from neon.experiments.experiment import Experiment
from ipdb import set_trace as trace


logger = logging.getLogger(__name__)


class HyperOpt(Experiment):
    """
    In this `Experiment`, a hyperyaml file is parsed to find parameter ranges,
    then
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


    def run(self):
        """
        Actually carry out each of the experiment steps.
        """
        pass

