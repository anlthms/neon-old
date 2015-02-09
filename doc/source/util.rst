.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Utility Functions
=================

Python 2 and 3 Compatibility
----------------------------

.. autosummary::
   :toctree: generated/

   neon.util.compat.PY3
   neon.util.compat.range

To ensure code runs under python2 and 3, you can utilize the definitions
in :mod:`neon.util.compat`

The :attr:`neon.util.compat.PY3` attribute will be set to True if we are
running under python3 and False otherwise.

The :attr:`neon.util.compat.range` shoud be used whenever a range of numbers is
needed.  On python2 this aliases ``xrange`` and under python3 it aliases
``range`` so in each case an iterator will be returned.  In situations where an
iterator is not feasible, wrapping the compatible range call in
``list(range(x))`` is recommended
