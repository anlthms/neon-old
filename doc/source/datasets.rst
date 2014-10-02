Datasets
========

Available Datasets
------------------

.. autosummary::
   :toctree: generated/

   neon.datasets.mnist.MNIST

Adding a new Dataset
--------------------

* Subclass :class:`neon.datasets.dataset.Dataset` ensuring to write an
  implementation of :func:`neon.datasets.dataset.Dataset.load`.
