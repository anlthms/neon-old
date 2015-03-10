# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Houses code for each of the core backend and associated Tensor data structures.
"""

import logging
import numpy as np
import os
import sys

from neon.backends.flexpt_dtype import flexpt

# import shortcuts
from neon.backends.cpu import CPU
from neon.backends.par import NoPar, ModelPar, DataPar

if np.__dict__.get('flexpt') is not None:
    raise RuntimeError('The numpy package already has a flexpt type')

np.flexpt = flexpt
np.typeDict['flexpt'] = np.dtype(flexpt)


def gen_backend(model, gpu=False, nrv=False, datapar=False, modelpar=False,
                flexpoint=False, rng_seed=None, numerr_handling=None,
                half=False, stochastic_round=0, device_id=None):
    """
    Construct and return a backend instance of the appropriate type based on
    the arguments given.  With no parameters, a single CPU core, float32
    backend is returned.

    Arguments:
        model (neon.models.model.Model): The instantiated model upon which we
                                         will utilize this backend.
        gpu (bool, optional): If True, attempt to utilize a CUDA capable GPU if
                              installed in the system.  Defaults to False which
                              implies a CPU based backend.
        nrv (bool, optional): If True, attempt to utilize the Nervana Engine
                              for computation (must be installed on the
                              system).  Defaults to False which implies a CPU
                              based backend.
        datapar (bool, optional): Set to True to ensure that data is
                                  partitioned and each chunk is processed in
                                  parallel on different compute cores. Requires
                                  mpi4py.  Defaults to False which implies that
                                  all data will be processed sequentially on a
                                  single compute core.
        modelpar (bool, optional): Set to True to ensure that the nodes in each
                                   model layer are partitioned and distributed
                                   across multiple compute cores.  Requires
                                   mpi4py.  Defaults to False which implies
                                   that all nodes in all model layers will be
                                   processed by the same single compute core.
        flexpoint (bool, optional): If True, attempt to use FlexPoint(TM)
                                    element typed data instead of the default
                                    float32 which is in place if set to False.
        rng_seed (numeric, optional): Set this to a numeric value which can be
                                      used to seed the random number generator
                                      of the instantiated backend.  Defaults to
                                      None, which doesn't explicitly seed (so
                                      each run will be different)
        stochastic_round (numeric, optional): Only affects the max backend. If
                                              1, perform stochastic rounding.
                                              If 0, round to nearest.
        numerr_handling (dict, optional): Dictate how numeric errors are
                                          displayed and handled.  The keys and
                                          values permissible for this dict
                                          match that seen in numpy.seterr.
                                          If set to None (the default),
                                          behavior is equivalent to
                                          {'all': 'warn'}
        device_id (numeric, optional): Set this to a numeric value which can be
                                       used to select which device to run the
                                       process on

    Returns:
        Backend: newly constructed backend instance of the specifed type.

    Notes:
        * Attempts to construct a GPU instance without a CUDA capable card or
          without Nervana's cuda-convnet2 based cudanet package will resort to
          a CPU instance with a warning generated.
        * Attempts to construct a parallel instance without mpi4py installed
          will resort in a non-distributed backend instance with a warning
          generated.
        * The returned backend will still need to call its par.init_model()
          at some point after the model has been linked, in order for parallel
          training to proceed.
    """
    logger = logging.getLogger(__name__)

    if gpu:
        gpu = False
        if sys.platform.startswith("linux"):
            gpu = (os.system("nvidia-smi > /dev/null 2>&1") == 0)
        elif sys.platform.startswith("darwin"):
            gpu = (os.system("kextstat | grep -i cuda > /dev/null 2>&1") == 0)
        if gpu:
            try:
                import cudanet  # noqa
            except ImportError:
                logger.warning("cudanet not found, can't run via GPU")
                gpu = False
        else:
            logger.warning("Can't find CUDA capable GPU")
    elif nrv:
        nrv = False
        try:
            from umd.nrv_backend import NRVBackend
            nrv = True
        except ImportError:
            logger.warning("Nervana Engine system software not found")

    if flexpoint:
        logger.warning("Flexpoint(TM) backend not currently available")

    if datapar and modelpar:
        raise NotImplementedError('Hybrid parallelization scheme not '
                                  'implemented yet.  Try with at most one of'
                                  'datapar or modelpar')
    if modelpar:
        par = ModelPar()
    elif datapar:
        par = DataPar()
    else:
        par = NoPar()

    if par.device_id is not None:
        if device_id is not None:
            logger.warn('Ignoring device id specified in command line.')
        device_id = par.device_id

    if gpu:
        from neon.backends.gpu import GPU
        be_name = 'GPU'
        be = GPU(rng_seed=rng_seed, device_id=device_id)
    elif half:
        import pycuda.autoinit  # create the context
        from neon.backends.max import MAX
        be_name = 'MAX_FP16'
        be = MAX(rng_seed=rng_seed, stochastic_round=stochastic_round,
                 device_id=device_id)
    elif nrv:
        be_name = 'NRV'
        be = NRVBackend(rng_seed=rng_seed, seterr_handling=numerr_handling,
                        device_id=device_id)
    else:
        be_name = 'CPU'
        be = CPU(rng_seed=rng_seed, seterr_handling=numerr_handling)
    logger.info("{} backend, RNG seed: {}, numerr: {}".format
                (be_name, rng_seed, numerr_handling))

    par.associate(be)
    return be
