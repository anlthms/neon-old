Backends
========

Current Implementations
-----------------------

.. autosummary::
   :toctree: generated/

   mylearn.backends._numpy.Numpy
   mylearn.backends._cudamat.Cudamat

Adding a new Backend
--------------------

* Generate a subclass of :class:`mylearn.backends.backend.Backend` including an
  associated tensor class :class:`mylearn.backends.backend.Backend.Tensor`.

* Implement overloaded operators to manipulate these tensor objects, as well
  other operations.

* To date, these operations have attempted to mimic numpy syntax.
