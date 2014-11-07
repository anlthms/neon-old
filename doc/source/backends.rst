Backends
========

Current Implementations
-----------------------

.. autosummary::
   :toctree: generated/

   neon.backends.cpu.CPU
   neon.backends.gpu.GPU
   neon.backends.flexpoint.Flexpoint

   neon.backends.unsupported._numpy.Numpy
   neon.backends.unsupported._cudamat.Cudamat
   neon.backends.unsupported._cudanet.Cudanet

Adding a new Backend
--------------------

1. Generate a subclass of :class:`neon.backends.backend.Backend` including an
   associated tensor class :class:`neon.backends.backend.Backend.Tensor`.

2. Implement overloaded operators to manipulate these tensor objects, as well
   other operations.

3. To date, these operations have attempted to mimic numpy syntax.
