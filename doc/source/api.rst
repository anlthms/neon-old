.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------
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

Cudanet GPU
-----------

.. autosummary::
   :toctree: generated/

   neon.backends.cc2.GPU

Nervana GPU
-----------

.. autosummary::
  :toctree: generated/

  neon.backends.gpu.GPU

Flexpoint™
----------

.. autosummary::
   :toctree: generated/

   neon.backends.flexpoint.Flexpoint

Nervana Hardware
----------------

To add


Models
======

.. autosummary::
   :toctree: generated/

   neon.models.model.Model

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

Balance Network
---------------

.. autosummary::
   :toctree: generated/

   neon.models.balance.Balance

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

Recurrent Neural Network
------------------------

.. autosummary::
   :toctree: generated/

   neon.models.rnn.RNN


Layers
======

.. autosummary::
   :toctree: generated/

   neon.layers.layer.Layer

Cost Layer
----------

.. autosummary::
   :toctree: generated/

   neon.layers.layer.CostLayer

Activation Layer
----------------

.. autosummary::
   :toctree: generated/

   neon.layers.layer.ActivationLayer

Data Layer
----------

.. autosummary::
   :toctree: generated/

   neon.layers.layer.DataLayer

Weight Layer
------------

.. autosummary::
   :toctree: generated/

   neon.layers.layer.WeightLayer

Fully Connected Layer
---------------------

.. autosummary::
   :toctree: generated/

   neon.layers.fully_connected.FCLayer

Convolutional Layer
-------------------

.. autosummary::
   :toctree: generated/

   neon.layers.convolutional.ConvLayer

Pooling Layers
---------------

.. autosummary::
   :toctree: generated/

   neon.layers.pooling.PoolingLayer
   neon.layers.pooling.CrossMapPoolingLayer

DropOut Layer
-------------

.. autosummary::
   :toctree: generated/

   neon.layers.dropout.DropOutLayer

Composite Layers
----------------

.. autosummary::
   :toctree: generated/

   neon.layers.compositional.CompositeLayer
   neon.layers.compositional.BranchLayer
   neon.layers.compositional.ListLayer

Normalized Layers
-----------------

.. autosummary::
   :toctree: generated/

   neon.layers.normalizing.CrossMapResponseNormLayer
   neon.layers.normalizing.LocalContrastNormLayer

Recurrent Layers
----------------

.. autosummary::
   :toctree: generated/

   neon.layers.recurrent.RecurrentLayer
   neon.layers.recurrent.RecurrentCostLayer
   neon.layers.recurrent.RecurrentHiddenLayer
   neon.layers.recurrent.RecurrentOutputLayer
   neon.layers.recurrent.RecurrentLSTMLayer


Learning Rules
==============

.. autosummary::
   :toctree: generated/

   neon.optimizers.learning_rule.LearningRule

Gradient Descent
----------------

.. autosummary::
   :toctree: generated/

   neon.optimizers.gradient_descent.GradientDescent
   neon.optimizers.gradient_descent.GradientDescentPretrain
   neon.optimizers.gradient_descent.GradientDescentMomentum
   neon.optimizers.gradient_descent.GradientDescentMomentumWeightDecay
   neon.optimizers.adadelta.AdaDelta

Parameter Related
=================

Value Initialization
--------------------

.. autosummary::
   :toctree: generated/

   neon.params.val_init.UniformValGen
   neon.params.val_init.AutoUniformValGen
   neon.params.val_init.GaussianValGen
   neon.params.val_init.SparseEigenValGen
   neon.params.val_init.NodeNormalizedValGen


Transforms
==========

Activation Functions
--------------------

.. autosummary::
   :toctree: generated/

   neon.transforms.rectified.RectLin
   neon.transforms.rectified.RectLeaky
   neon.transforms.logistic.Logistic
   neon.transforms.tanh.Tanh
   neon.transforms.softmax.Softmax

Cost Functions
--------------

.. autosummary::
   :toctree: generated/

   neon.transforms.sum_squared.SumSquaredDiffs
   neon.transforms.cross_entropy.CrossEntropy
   neon.transforms.xcov.XCovariance


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
---------

.. autosummary::
   :toctree: generated/

   neon.datasets.sparsenet.SPARSENET

ImageNet
--------

.. autosummary::
   :toctree: generated/

   neon.datasets.i1k.I1K

Mobydick
--------

.. autosummary::
   :toctree: generated/

   neon.datasets.mobydick.MOBYDICK

Imageset
--------

.. autosummary::
   :toctree: generated/

   neon.datasets.imageset.Imageset

Synthetic
---------

.. autosummary::
   :toctree: generated/

   neon.datasets.synthetic.UniformRandom
   neon.datasets.synthetic.ToyImages


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
   neon.util.compat.range
   neon.util.compat.StringIO
