Utility Functions
=================

Python 2 and 3 Compatibility
----------------------------

.. autosummary::
   :toctree: generated/

   neon.util.compat.PY3

To ensure code runs under python2 and 3, you can utilize the definitions
in :mod:`neon.util.compat`

The :attr:`neon.util.compat.PY3` attribute will be set to True if we are
running under python3 and False otherwise.


CUDA GPU Compatibility
----------------------

.. autosummary::
   :toctree: generated/

   neon.util.compat.CUDA_GPU

To conditionally run code on machines with CUDA compatible GPU's installed, you
can utilize the :attr:`neon.util.compat.CUDA_GPU`.  It will be set to True if
such a GPU is installed, and False otherwise.


Distributed System Compatibility
--------------------------------

.. autosummary::
   :toctree: generated/

   neon.util.compat.MPI_INSTALLED

To conditionally run code on machines configured to run on multiple
cores/machines in parallel via OpenMPI and mpi4py, you
can utilize the :attr:`neon.util.compat.MPI_INSTALLED`.  It will be set to
True if such a setup is configured, and False otherwise.
