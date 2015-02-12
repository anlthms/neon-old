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
   neon.util.compat.StringIO

To ensure code runs under python2 and 3, you can utilize the definitions
in :mod:`neon.util.compat`

The :attr:`neon.util.compat.PY3` attribute will be set to True if we are
running under python3 and False otherwise.

The :attr:`neon.util.compat.range` shoud be used whenever a range of numbers is
needed.  On python2 this aliases ``xrange`` and under python3 it aliases
``range`` so in each case an iterator will be returned.  In situations where an
iterator is not feasible, wrapping the compatible range call in
``list(range(x))`` is recommended

The :attr:`neon.util.compat.pickle` should be used for serialization and
deserialization of python objects.  On python2 it aliases cPickle and under
python3 it aliases pickle (which first attempts to import the faster cPickle
equivalent where available).

The :attr:`neon.util.compat.queue` should be used whenever the python2 Queue
module is needed.  In python3 this was renamed to queue.

The :attr:`neon.util.compat.StringIO` should be used whenever you need to read
and write strings as files.
