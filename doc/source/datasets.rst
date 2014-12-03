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
   neon.datasets.synthetic.UniformRandom

Adding a new Dataset
--------------------

* Subclass :class:`neon.datasets.dataset.Dataset` ensuring to write an
  implementation of :func:`neon.datasets.dataset.Dataset.load`.
