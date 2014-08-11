Transforms
==========

Transforms are functions that can be applied to modify data values.
Often these will represent things like non-linear activation functions, or
cost/loss functions.


Available Transforms
--------------------

.. autosummary::
   :toctree: generated/

   mylearn.transforms.linear.Identity
   mylearn.transforms.rectified.RectLin
   mylearn.transforms.logistic.Logistic
   mylearn.transforms.logistic.PseudoLogistic
   mylearn.transforms.tanh.Tanh

   mylearn.transforms.sum_squared.SumSquaredDiffs
   mylearn.transforms.cross_entropy.CrossEntropy
