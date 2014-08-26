"""
Houses code for each of the core backend and associated Tensor data structures.
"""

import numpy as np

from mylearn.backends.fixpt_dtype import fixpt

if np.__dict__.get('fixpt') is not None:
    raise RuntimeError('The numpy package already has a fixpt type')

np.fixpt = fixpt
np.typeDict['fixpt'] = np.dtype(fixpt)
