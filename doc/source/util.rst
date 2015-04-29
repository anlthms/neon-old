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


Persistence of objects and data
-------------------------------

.. autosummary::
   :toctree: generated/

   neon.util.persist.ensure_dirs_exist
   neon.util.persist.deserialize
   neon.util.persist.serialize
   neon.util.persist.YAMLable

To save and load python objects to and from disk, you'll want to make use of
:func:`neon.util.persist.serialize` and :func:`neon.util.persist.deserialize`
respectively.  For python objects we tend to make use of python's built-in
pickle (.pkl) file format.

:func:`neon.util.persist.deserialize` is also used to parse our input YAML
files.

To ensure that any new type of python object can be understood when listed in
YAML file format, it should be a subclass of
:class:`neon.util.persist.YAMLable`.


Batched Data Writing
--------------------

.. autosummary::
   :toctree: generated/

   neon.util.batch_writer.BatchWriter
   neon.util.batch_writer.BatchWriterImagenet

The above class can be used to efficiently pipeline and load a particular
dataset in a batched fashion.

Included is an implemented example tailored for the :class:`neon.datasets.I1K`
(Imagenet) dataset adapted from cuda-convnet2 code.

Note that these classes required the installation of imgworker, an optional
dependency described in :doc:`installation`
