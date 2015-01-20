.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Datasets
========

Available Datasets
------------------

.. autosummary::
   :toctree: generated/

   neon.datasets.cifar10.CIFAR10
   neon.datasets.iris.Iris
   neon.datasets.mnist.MNIST
   neon.datasets.sparsenet.SPARSENET
   neon.datasets.i1k.I1K
   neon.datasets.ndsb.NDSB
   neon.datasets.mobydick.MOBYDICK
   neon.datasets.tfd.TFD
   neon.datasets.synthetic.UniformRandom
   neon.datasets.synthetic.ToyImages

Adding a new Dataset
--------------------

* Subclass :class:`neon.datasets.dataset.Dataset` ensuring to write an
  implementation of :func:`neon.datasets.dataset.Dataset.load`.
