Models
======


Available Models
----------------

.. autosummary::
   :toctree: generated/

   neon.models.mlp.MLP
   neon.models.autoencoder.Autoencoder

Adding a new Model type
-----------------------

* Create a new subclass of :class:`neon.models.model.Model`

  * At a minimum implements :func:`neon.models.model.Model.fit` to learn
    parameters from a training dataset
  * :func:`neon.models.model.Model.predict` to apply learned parameters
    to make predictions about another dataset.

Adding a new Layer type
-----------------------

Neural network models are typically composed of several of these objects.

* Create a new subclass of :class:`neon.models.layer.Layer` to suit your
  needs.
