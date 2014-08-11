.. currentmodule:: mylearn
.. _api:

*************
API Reference 
*************

.. _api.functions:

Backends
========

.. autosummary::
   :toctree: generated/

   mylearn.backends.backend.Backend
..   mylearn.backends.backend.Backend.Tensor

Numpy
-----

.. autosummary::
   :toctree: generated/

   mylearn.backends._numpy.Numpy
..   mylearn.backends._numpy.Numpy.Tensor

CUDAMat
-------

.. autosummary::
   :toctree: generated/

   mylearn.backends._cudamat.Cudamat
..   mylearn.backends._cudamat.Cudamat.Tensor


Models
======

.. autosummary::
   :toctree: generated/

   mylearn.models.model.Model
   mylearn.models.layer.Layer

MLP
---

.. autosummary::
   :toctree: generated/

   mylearn.models.mlp.MLP

Autoencoder
-----------

.. autosummary::
   :toctree: generated/

   mylearn.models.autoencoder.Autoencoder


Transforms
==========

Activation Functions
--------------------

.. autosummary::
   :toctree: generated/

   mylearn.transforms.linear.Identity
   mylearn.transforms.rectified.RectLin
   mylearn.transforms.logistic.Logistic
   mylearn.transforms.logistic.PseudoLogistic
   mylearn.transforms.tanh.Tanh

Cost Functions
--------------

.. autosummary::
   :toctree: generated/

   mylearn.transforms.sum_squared.SumSquaredDiffs
   mylearn.transforms.cross_entropy.CrossEntropy


Datasets
========

.. autosummary::
   :toctree: generated/

   mylearn.datasets.dataset.Dataset

MNIST
-----

.. autosummary::
   :toctree: generated/

   mylearn.datasets.mnist.MNIST


Experiments
===========

.. autosummary::
   :toctree: generated/

   mylearn.experiments.experiment.Experiment


Miscellaneous
=============

.. autosummary::
   :toctree: generated/

   mylearn.util.compat
   mylearn.util.factory.Factory
