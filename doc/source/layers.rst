.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Layers
======

Neural network models are typically composed of a data layer, several weight
layers and a cost layer. Data layers are associated with a
:doc:`data set <datasets>`, and cost layers with a
:doc:`cost function<transforms>`. In addition to standard weight layers, there
is a variety of model specific weight, pooling and normalization layers.


Available Layers
----------------

.. autosummary::
   :toctree: generated/

   neon.layers.layer.Layer
   neon.layers.layer.CostLayer
   neon.layers.layer.DataLayer
   neon.layers.layer.ActivationLayer
   neon.layers.layer.WeightLayer


   neon.layers.compositional.CompositeLayer
   neon.layers.compositional.BranchLayer
   neon.layers.compositional.ListLayer

   neon.layers.convolutional.ConvLayer

   neon.layers.dropout.DropOutLayer
   neon.layers.fully_connected.FCLayer
   neon.layers.normalizing.CrossMapResponseNormLayer
   neon.layers.normalizing.LocalContrastNormLayer
   neon.layers.pooling.PoolingLayer
   neon.layers.pooling.CrossMapPoolingLayer

   neon.layers.recurrent.RecurrentLayer
   neon.layers.recurrent.RecurrentCostLayer
   neon.layers.recurrent.RecurrentOutputLayer
   neon.layers.recurrent.RecurrentHiddenLayer
   neon.layers.recurrent.RecurrentLSTMLayer


.. _extending_layer:

Adding a new type of Layer
--------------------------

#. Create a new subclass of :class:`neon.models.layer.Layer` to suit your
   needs.
#. provide implementation of functions: :func:`neon.layers.layer.Layer.fprop`
   :func:`neon.layers.layer.Layer.bprop` at a minimum.
