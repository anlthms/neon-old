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

CPU
---

.. autosummary::
   :toctree: generated/

   neon.backends.cpu.CPU

GPU
---

.. autosummary::
   :toctree: generated/

   neon.backends.gpu.GPU

Flexpoint
---------

.. autosummary::
   :toctree: generated/

   neon.backends.flexpoint.Flexpoint

Unsupported
------------

.. autosummary::
   :toctree: generated/

   neon.backends.unsupported._numpy.Numpy
   neon.backends.unsupported._cudamat.Cudamat
   neon.backends.unsupported._cudanet.Cudanet


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

RBM
---

.. autosummary::
   :toctree: generated/

   neon.models.rbm.RBM

DBN
---

.. autosummary::
   :toctree: generated/

   neon.models.dbn.DBN

GB
--

.. autosummary::
   :toctree: generated/

   neon.models.gb.GB


Transforms
==========

Activation Functions
--------------------

.. autosummary::
   :toctree: generated/

   neon.transforms.linear.Identity
   neon.transforms.rectified.RectLin
   neon.transforms.logistic.Logistic
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

CIFAR10
-------

.. autosummary::
   :toctree: generated/

   neon.datasets.cifar10.CIFAR10

Iris
----

.. autosummary::
   :toctree: generated/

   neon.datasets.iris.Iris

Sparsenet
----------

.. autosummary::
   :toctree: generated/

   neon.datasets.sparsenet.SPARSENET

Synthetic
---------

.. autosummary::
   :toctree: generated/

   neon.datasets.synthetic.UniformRandom


Experiments
===========

.. autosummary::
   :toctree: generated/

   neon.experiments.experiment.Experiment
   neon.experiments.fit.FitExperiment
   neon.experiments.fit_predict_err.FitPredictErrorExperiment
   neon.experiments.check_grad.GradientChecker


Miscellaneous
=============

.. autosummary::
   :toctree: generated/

   neon.util.compat.PY3
   neon.util.compat.CUDA_GPU
   neon.util.compat.MPI_INSTALLED
