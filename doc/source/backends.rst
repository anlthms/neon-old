.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Backends
========

Backends incorporate a basic multi-dimensional
:class:`neon.backends.backend.Backend.Tensor` data structure as well the
algebraic and deep learning specific operations that can be performed on them.

Each implemented backend conforms to our :doc:`ml_operational_layer` API to
ensure a consistent behaviour.

Current Implementations
-----------------------

.. autosummary::
   :toctree: generated/

   neon.backends.cpu.CPU
   neon.backends.gpu.GPU
   neon.backends.cc2.GPU
   neon.backends.par.NoPar
   neon.backends.par.DataPar
   neon.backends.par.ModelPar

Adding a new Backend
--------------------

1. Generate a subclass of :class:`neon.backends.backend.Backend` including an
   associated tensor class :class:`neon.backends.backend.Backend.Tensor`.

2. Implement overloaded operators to manipulate these tensor objects, as well
   other operations.  Effectively this amounts to implementing our MOP API (see
   :doc:`ml_operational_layer`)

3. To date, these operations have attempted to more or less mimic numpy syntax
   as much as possible.
