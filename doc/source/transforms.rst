Transforms
==========

Transforms are functions that can be applied to modify data values.
Often these will represent things like non-linear activation functions, or
cost/loss functions.


Available Transforms
--------------------

.. autosummary::
   :toctree: generated/

   neon.transforms.linear.Identity
   neon.transforms.rectified.RectLin
   neon.transforms.logistic.Logistic
   neon.transforms.logistic.PseudoLogistic
   neon.transforms.tanh.Tanh

   neon.transforms.sum_squared.SumSquaredDiffs
   neon.transforms.cross_entropy.CrossEntropy
