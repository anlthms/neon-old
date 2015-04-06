.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Datasets
========

Available Datasets
------------------

.. autosummary::
   :toctree: generated/

   neon.datasets.cifar10.CIFAR10
   neon.datasets.iris.Iris
   neon.datasets.mnist.MNIST
   neon.datasets.sparsenet.SPARSENET
   neon.datasets.i1k.I1K
   neon.datasets.imageset.Imageset
   neon.datasets.mobydick.MOBYDICK
   neon.datasets.synthetic.UniformRandom
   neon.datasets.synthetic.ToyImages

Adding a new Dataset
--------------------

* Subclass :class:`neon.datasets.dataset.Dataset` ensuring to write an
  implementation of :func:`neon.datasets.dataset.Dataset.load`.
* Datasets should have a single data point per row, and should either be in
  numpy ndarray format, or batched as such.
* Datasets are loaded and transformed by the approproate backend via the
  :func:`neon.datasets.dataset.Dataset.format` call.
* If you have image data, have a look at the
  :class:`neon.datasets.imageset.Imageset` and instructions for working with it
  desribed below.

Working with Imageset
---------------------
If you have a set of image files as input, consider using Imageset.  This
Dataset incorporates batching and pre-processing (cropping, normalization) in
an efficient manner.  It can also take advantage of directory subfolders to
identify target labels.

Required Imageset constructor parameters:

* batch_dir: where to keep batched data objects and indices
* image_dir: where the raw image files live
* macro_size: number of images to include in each macro batch
* cropped_image_size: desired number of pixels along 1 dimension
                      (assumes square images)
* output_image_size: original image number of pixels along 1 dimension
                     (assumes square images)

Optional Imageset parameters (mostly BatchWriter related):

* square_crop: make cropped image square
* zero_center: pixel intensities are divided by 128 to lie centered around 0
               (lie in range [-1, 1]).  Otherwise will lie in range [0, 1]
* tdims: number of dimensions of each target.
* label_list: array of label names
* num_channels: number of image channels (ex. 3 for RGB images)
* num_workers: number of processes to spawn for batch writing
* backend_type: element value type (for each image pixel)

