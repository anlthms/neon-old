.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Transforms
==========

Transforms are functions that can be applied to modify data values.
Often these will represent things like non-linear activation functions, or
cost/loss functions.


Available Transforms
--------------------

.. autosummary::
   :toctree: generated/

   neon.transforms.linear.Linear
   neon.transforms.rectified.RectLin
   neon.transforms.rectified.RectLeaky
   neon.transforms.logistic.Logistic
   neon.transforms.tanh.Tanh
   neon.transforms.softmax.Softmax
   neon.transforms.batch_norm.BatchNorm

   neon.transforms.sum_squared.SumSquaredDiffs
   neon.transforms.cross_entropy.CrossEntropy
   neon.transforms.xcov.XCovariance
