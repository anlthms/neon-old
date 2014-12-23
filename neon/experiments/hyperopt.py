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

def parse_line(line):
    """
    move this function into hyperopt
    """
    # example: 0                  1              2         3       4
    # ['learning_rate:', '!hyperopt:shotgun', '{float,', '0.1,', '1.0},', '#', 'range', 'for', 'HYPEROPT.']
    dic = [k.strip("{},") for k in line.split()]
    # check sanity
    if dic[2] != 'float' and dic[2] != 'int' and dic[2] != 'cat':
        raise NameError('BadParameterError')

    ho = dict()
    ho['chooser'] = dic[1].split(':')[1]
    ho['type'] = dic[2]
    ho['end'] = float(dic[3])
    ho['start'] = float(dic[4])

    out = get_parameters(ho)

    return dic[0]+" "+str(out)+",\n"

def get_parameters(ho):
    """
    cast range to number:
    this function is a placeholder for calls to spearmint

    (this should be wrapped in a class at some point)
    """
    if ho['chooser'] == 'shotgun':
        out = (ho['end'] - ho['start']) * np.random.rand() + ho['start']
        print "----BURNING IN PARAMETERS", out, "-------"
    else:
        out = None
        raise NotImplementedError('Unknown chooser for hyperpotimization')
    return out

