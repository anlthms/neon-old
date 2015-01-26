# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Houses code for each of the core backend and associated Tensor data structures.
"""

import numpy as np

from neon.backends.flexpt_dtype import flexpt

# import shortcuts
from neon.backends.cpu import CPU  # noqa
from neon.backends.gpu import GPU  # noqa

if np.__dict__.get('flexpt') is not None:
    raise RuntimeError('The numpy package already has a flexpt type')

np.flexpt = flexpt
np.typeDict['flexpt'] = np.dtype(flexpt)
