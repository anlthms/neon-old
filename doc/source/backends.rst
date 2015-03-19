.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Backends
========

Current Implementations
-----------------------

.. autosummary::
   :toctree: generated/

   neon.backends.cpu.CPU
   neon.backends.gpu.GPU
   neon.backends.flexpoint.Flexpoint
   neon.backends.par.NoPar
   neon.backends.par.DataPar
   neon.backends.par.ModelPar
..   neon.backends.max.MAX

Adding a new Backend
--------------------

1. Generate a subclass of :class:`neon.backends.backend.Backend` including an
   associated tensor class :class:`neon.backends.backend.Backend.Tensor`.

2. Implement overloaded operators to manipulate these tensor objects, as well
   other operations.

3. To date, these operations have attempted to mimic numpy syntax.
