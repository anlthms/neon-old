# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains various functions and wrappers to make code python 2 and python 3
compatible, as well as test for the presence of a CUDA compatible GPU (or at
least the CUDA SDK), and MPI for distributed networks.
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
mpi_size = 1
mpi_rank = 0
try:
    from mpi4py import MPI  # flake8: noqa
    MPI_INSTALLED = True
    mpi_size = MPI.COMM_WORLD.size
    mpi_rank = MPI.COMM_WORLD.rank
except ImportError:
    logger.warning('mpi4py not found')

# keep range calls consistent between python 2 and 3
# note: if you need a list and not an iterator you can do list(range(x))
range = range
if not PY3:
    logger.info("using xrange as range")
    range = xrange


def generate_backend(gpu=False, parallel=False, flexpoint=False,
                     rng_seed=None, numerr_handling=None):
    """
    Construct and return a backend instance of the appropriate type based on
    the arguments given.  With no parameters, a single CPU core, float32
    backend is returned.

    Arguments:
        gpu (bool, optional): If True, attempt to utilize a CUDA capable GPU if
                              installed in the system.  Defaults to False.
        parallel (bool, optional): If True, attempt to utilize mpi4py to
                                   construct a distributed backend.  Defaults
                                   to False.
        flexpoint (bool, optional): If True, attempt use FlexPoint(TM) element
                                    typed data instead of the default float32
                                    which is in place if set to False.
        rng_seed (numeric, optional): Set this to a numeric value which can be
                                      used to seed the random number generator
                                      of the instantiated backend.  Defaults to
                                      None, which doesn't explicitly seed.
        numerr_handling (dict, optional): Dictate how numeric errors are
                                          displayed and handled.  The keys and
                                          values permissible for this dict
                                          match that seen in numpy.seterr.

    Returns:
        Backend: newly constructed backend instance of the specifed type.

    Notes:
        Attempts to construct a GPU instance without a CUDA capable card will
        resort to a CPU instance with a warning generated.
        Attempts to construct a parallel instance without mpi4py installed will
        resort to a non-distributed backend instance with a warning generated.
    """
    compute = "CPU"
    num_cores = 1
    dtype = "float32"

    if gpu:
        gpu = False
        if sys.platform.startswith("linux"):
            gpu = (os.system("nvidia-smi > /dev/null 2>&1") == 0)
        elif sys.platform.startswith("darwin"):
            gpu = (os.system("kextstat | grep -i cuda > /dev/null 2>&1") == 0)
        if gpu:
            try:
                import cudanet  # noqa
                compute = "GPU"
            except ImportError:
                logger.warning("cudanet not found, can't run via GPU")
        else:
            logger.warning("Can't find CUDA capable GPU")

    if parallel:
        try:
            import mpi4py  # noqa
            num_cores = 2
        except ImportError:
            logger.warning("mpi4py not found, can't run distributed")

    if flexpoint:
        logger.warning("Flexpoint(TM) backend not currently available")


    # TODO: add flexpoint, distributed backends
    if compute == "GPU":
        from neon.backends.gpu import GPU
        be = GPU(rng_seed=rng_seed)
    else:
        from neon.backends.cpu import CPU
        be = CPU(rng_seed=rng_seed, seterr_handling=numerr_handling)
    return be
