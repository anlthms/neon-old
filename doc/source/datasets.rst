Datasets
========

Available Datasets
------------------

.. autosummary::
   :toctree: generated/

   mylearn.datasets.mnist.MNIST

Adding a new Dataset
--------------------

* Subclass :class:`mylearn.datasets.dataset.Dataset` ensuring to write an
  implementation of :func:`mylearn.datasets.dataset.Dataset.load`.
