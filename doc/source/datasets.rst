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
   neon.datasets.mobydick.MOBYDICK
   neon.datasets.tfd.TFD
   neon.datasets.synthetic.UniformRandom
   neon.datasets.synthetic.ToyImages

.. _extending_dataset:

Adding a new type of Dataset
----------------------------

#. Subclass :class:`neon.datasets.dataset.Dataset` 
#. write an implementation of :func:`neon.datasets.dataset.Dataset.load`.

* Datasets should have a single data point per row, and should either be in
  numpy ndarray format, or batched as such.
* Datasets are loaded and transformed by the appropriate backend via the
  :func:`neon.datasets.dataset.Dataset.format` call.
