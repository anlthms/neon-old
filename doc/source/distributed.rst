.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Distributed Implementations using MPI
=====================================


Available Models
----------------

Existing Models and Datasets can be parallelized by adding the ``--datapar`` or
``--modelpar`` command line parameters.

In the ``--datapar`` (data parallel) approach, data examples are partitioned
and distributed across multiple processes.  A separate model replica lives on
each process, and parameter values are synchronized across the models
to ensure each replica remains (evntually) consistent.

In the ``--modelpar`` (model parallel) approach, layer nodes are partitioned
and distributed across multiple processes.  Activations are then communicated
between processes whose nodes are connected.  At this time, we support model
parallelism on fully connected model layers only.

Parameter server based asynchronous SGD is not yet implemented, but please
contact us if this is something you need for your use case.
