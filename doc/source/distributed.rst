.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Distributed Implementations using MPI
=====================================


Available Models
----------------

.. autosummary::
   :toctree: generated/

   neon.models.convnet_dist.ConvnetDist
   neon.models.gb_dist.GBDist

Existing Models and Datasets can be parallelized by adding the ``--datapar`` or
``--modelpar`` command line parameters.

We support 3 typical ways in which the dataset can be parallelized to be fed
into the input layer of a deep neural network (parameter: dist_mode). 

1. Halo/tower approach (halopar): Most relevant for images or other 2-D data
   with convolutional style features. Inspired by the Coates et al.
   "Deep Learning with COTS HPC" paper. We are using the halo/tower approach
   for distributing the convnet and the sparse autoencoder (gb). 
2. Treat as a 1-d vector split amongst nodes (modelpar): Most relevant for
   fully connected MLP layers, but applicable to flattened images or 2-D
   datasets also.
3. Data parallel (datapar): Send a different micro-batch to each process.
   Useful for completely data parallel implementations. 
4. Combination of halo/tower parallel (each node gets a different 'quad'-rant
   of the image), and data parallel (each process within the node gets a
   different micro-batch, but micro-batches are aligned across nodes)
   (halodatapar). 

There is an interaction between the strategy used to parallelize the dataset
and the corresponding Layer type used.

1. halopar for dataset can be used with:

  a. ConvLayerDist as input layer (and MaxPoolingLayerDist as higher-level
     layer)
  b. LocalFilteringLayerDist as input layer (and L2PoolingLayerDist and
     LCNLayerDist as higher-level layers)

2. vecpar for dataset can be used with LayerDist and LayerWithNoBiasDist
3. datapar can be used with any layer, as long as the model or layer bprop
   supports a dist_mode='datapar'. For MLP (incl. CNN) this is implemented by
   subclassing the CPU backend in the CPUDataDist backend and extending the
   update_fc_dot() function.

Halopar and vecpar are recommended when working with large images or feature
vectors that can be split across processes. ConvnetDist and GBDist use halopar
for the convolutional-style layers and vecpar for the fully connected layers.
MLPDist uses vecpar. Datapar is recommended when working with large mini-batch
sizes that can be split across processes. Datapar for MLP can be enabled by
setting dist_flag=True in the YAML file and setting dist_mode='datapar'.

Parameter server based asynchronous SGD is not yet implemented, but please
contact us if this is something you need for your use case.

For distributing a new Model, Layer or Dataset using **halopar** follow the
recipe below:

Distributing a new Model
------------------------

1. Implement the Distributed Model as a derived class of the corresponding
   non-distributed Model class.
2. To build a Distributed Model where layers have halo terms, associate with
   the layers a .input object of type GlobalArray that is used for halo
   transfers during fprop and bprop and is used to create the fprop and bprop
   views. See adjust_for_dist() functions in ConvnetDist and GBDist models for
   examples of .input object creation. <model>.adjust_for_dist() functions
   should also call layer.adjust_for_dist() that adjust the layer.ifmshape and
   related matrix sizes (e.g. weight matrix).
3. In general, combining different types of layers (conv style, FC; padded vs
   not padded) is tricky and needs to be thought through for its impact on
   <model>.fprop and <model>.bprop. Contact arjun@nervanasys.com for
   assistance.
4. Follow steps below for distributing associated Layer and Dataset classes.

Distributing a new Layer
------------------------

1. Implement the Distributed Layer as a derived class of the non-distributed
   Layer class.
2. At a high-level user has to decide whether fprop and bprop for the current
   layer expect halo transfers before hand or not. This can depend on what
   type of layer it is (is it convolutional or not)? If the input is not
   already halo consistent before hand, then get_fprop_view() is called within
   the fprop function of the Layer. If the input is halo consistent, then
   there is no need for calling get_fprop_view() within fprop. For example, in
   L2PoolingLayerDist:

.. code-block:: bash

    def fprop(self, inputs_):
        inputs = self.input.get_fprop_view(inputs_)
        super(L2PoolingLayerDist, self).fprop(inputs)

    # Similarly for bprop
    def bprop(self, error, inputs_, epoch, momentum):
        # Assuming 'error' is already halo consistent
        # In GBDist model:
        # During unsupervised pre-training, 'error' does not have halos
        # During supervised training, 'error' does have halos...
        # ...L2PoolingLayerDist is connected to LCNDist Layers which...
        # ...take care of the halo transfers in bprop:
        # self.berror = (self.input.get_bprop_view(self.berror))
        
        # redo-ing get_fprop_view for inputs, could cache for speed-up
        inputs = self.input.get_fprop_view(inputs_)
        super(L2PoolingLayerDist, self).bprop(error, inputs, epoch, momentum)		


3. YAML file: Same steps as under YAML for distributing a new dataset below.


Distributing a new Dataset
--------------------------

* Changes in YAML file for distributing a new dataset X, assuming distributed
  Layers and Models have been generated for another dataset Y (and optionally
  dataset X has been trained before with non-distributed Layers and Models).
  See examples/mnist_distarray_*.yaml or examples/cifar10_distarray_*.yaml for
  details.

1. Add 'dist_flag: True' to experiments and datasets 
2. Change model name from <model_name.ModelName> to
   <model_name_dist.ModelNameDist>. For e.g.: gb.GB to gb_dist.GBDist
3. Add Dist suffix to Layer class names
4. Change dataset serialize name to include a {rank} and {size} parameter
5. If debugging and comparing accuracy with non-dist implementation, make sure
   filter and layer sizes etc. are the same as in non-dist code, because this
   could change the random number initialization between the dist and non-dist
   code. Obviously, the specific examples sampled for training also need to be
   the same. You might need to manually erase the previously pkl'd files and
   re-pkl them.


* Changes in Dataset class file (e.g. mnist.py or cifar10.py): Look at
  self.dist_flag in an existing dataset (e.g. MNIST) and add similar handling
  for new dataset:

1. In __init__(), make sure comm.size is handled. Currently require comm.size
   to be a square and divide image width and height. For MNIST (28x28) or
   CIFAR (32x32) only n=1, 4, or 16 make sense for now.
2. In read_image_file(): extract and return the correct ‘quad’ or n-rant of
   the image.
3. In load(): adjust the size of the array that will store the local n-rant
   of the image.
