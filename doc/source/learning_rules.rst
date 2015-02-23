.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Learning Rules
==============

Learning rules are optimizers that use backpropagated gradients to update
layer weights. The most basic form is stochastic gradient descent (SGD), which
can be augmented with momentum. AdaDelta (Zeiler 2012) is an adaptive gradient
method that does not require the leraning rate to be tuned manually.


Available Learning Rules
------------------------

.. autosummary::
   :toctree: generated/

   neon.optimizers.gradient_descent.GradientDescent
   neon.optimizers.gradient_descent.GradientDescentPretrain
   neon.optimizers.gradient_descent.GradientDescentMomentum
   neon.optimizers.gradient_descent.GradientDescentMomentumWeightDecay
   neon.optimizers.adadelta.AdaDelta

