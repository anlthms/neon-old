# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains various functions and wrappers to make code python 2 and python 3
compatible, as well as indicate the presence of a CUDA compatible GPU (or at
least the CUDA SDK).
"""

import os
import sys
import logging


logger = logging.getLogger(__name__)
PY3 = (sys.version_info[0] >= 3)

CUDA_GPU = False
if sys.platform.startswith("linux"):
    CUDA_GPU = (os.system("nvidia-smi > /dev/null 2>&1") == 0)
elif sys.platform.startswith("darwin"):
    CUDA_GPU = (os.system("kextstat | grep -i cuda > /dev/null 2>&1") == 0)
if CUDA_GPU:
    try:
        import cudanet
    except ImportError:
        logger.warning("cudanet not found, can't set CUDA_GPU")
        CUDA_GPU = False

MPI_INSTALLED = False
try:
    import mpi4py  # flake8: noqa
    MPI_INSTALLED = True
except ImportError:
    logger.warning('mpi4py not found')

# keep range calls consistent between python 2 and 3
# note: if you need a list and not an iterator you can do list(range(x))
range = range
if not PY3:
    logger.info("using xrange as range")
    range = xrange