.. currentmodule:: neon
.. _api:

*************
API Reference 
*************

.. _api.functions:

Backends
========

.. autosummary::
   :toctree: generated/

   neon.backends.backend.Backend
..   neon.backends.backend.Backend.Tensor

Numpy
-----

.. autosummary::
   :toctree: generated/

   neon.backends._numpy.Numpy
..   neon.backends._numpy.Numpy.Tensor

CUDAMat
-------

.. autosummary::
   :toctree: generated/

   neon.backends._cudamat.Cudamat
..   neon.backends._cudamat.Cudamat.Tensor


Models
======

.. autosummary::
   :toctree: generated/

   neon.models.model.Model
   neon.models.layer.Layer

MLP
---

.. autosummary::
   :toctree: generated/

   neon.models.mlp.MLP

Autoencoder
-----------

.. autosummary::
   :toctree: generated/

   neon.models.autoencoder.Autoencoder


Transforms
==========

Activation Functions
--------------------

.. autosummary::
   :toctree: generated/

   neon.transforms.linear.Identity
   neon.transforms.rectified.RectLin
   neon.transforms.logistic.Logistic
   neon.transforms.logistic.PseudoLogistic
   neon.transforms.tanh.Tanh

Cost Functions
--------------

.. autosummary::
   :toctree: generated/

   neon.transforms.sum_squared.SumSquaredDiffs
   neon.transforms.cross_entropy.CrossEntropy


Datasets
========

.. autosummary::
   :toctree: generated/

   neon.datasets.dataset.Dataset

MNIST
-----

.. autosummary::
   :toctree: generated/

   neon.datasets.mnist.MNIST


Experiments
===========

.. autosummary::
   :toctree: generated/

   neon.experiments.experiment.Experiment


Miscellaneous
=============

.. autosummary::
   :toctree: generated/

   neon.util.compat.PY3
   neon.util.factory.Factory
