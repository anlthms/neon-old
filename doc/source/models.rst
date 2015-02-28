.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Models
======


Available Models
----------------

.. autosummary::
   :toctree: generated/

   neon.models.mlp.MLP
   neon.models.autoencoder.Autoencoder
   neon.models.gb.GB
   neon.models.gb_dist.GBDist
   neon.models.rbm.RBM
   neon.models.dbn.DBN

.. _extending_model:

Adding a new type of Model
--------------------------

#. Create a new subclass of :class:`neon.models.model.Model`
#. At a minimum implement :func:`neon.models.model.Model.fit` to learn
   parameters from a training dataset
#. Write :func:`neon.models.model.Model.predict` to apply learned parameters
   to make predictions about another dataset.
