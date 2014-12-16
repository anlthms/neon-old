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
from ipdb import set_trace as trace


logger = logging.getLogger(__name__)


def get_parameters(res):
    """
    cast range to number:
    this function is a placeholder for calls to spearmint

    (this should be wrapped in a class at some point)
    """
    if res['chooser'] == 'shotgun':
        out = (res['end'] - res['start']) * np.random.rand() + res['start']
    else:
        out = None
        raise NotImplementedError('Unknown chooser for hyperpotimization')
    return out

