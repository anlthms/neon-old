Models
======


Available Models
----------------

.. autosummary::
   :toctree: generated/

   mylearn.models.mlp.MLP
   mylearn.models.autoencoder.Autoencoder

Adding a new Model type
-----------------------

* Create a new subclass of :class:`mylearn.models.model.Model`

  * At a minimum implements :func:`mylearn.models.model.Model.fit` to learn
    parameters from a training dataset
  * :func:`mylearn.models.model.Model.predict` to apply learned parameters
    to make predictions about another dataset.

Adding a new Layer type
-----------------------

Neural network models are typically composed of several of these objects.

* Create a new subclass of :class:`mylearn.models.layer.Layer` to suit your
  needs.
