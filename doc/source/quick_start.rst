.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Quick start
===========

On a Mac OSX or Linux box enter the following to download and
install neon, and use it to train your first multi-layer perceptron

.. code-block:: bash

    git clone https://github.com/NervanaSystems/neon.git
    cd neon
    sudo make install
    neon examples/mlp/mnist-small.yaml


Next Steps
----------

* Read the :doc:`installation` instructions to enable faster backends (if you
  have a GPU or Nervana Hardware), and configure other functionality
* See how to run neon and adjust various command line flags in :doc:`using_neon`
* Learn how to :ref:`train_models` effectively
* Understand the :doc:`api` and architecture of the neon codebase.
* Contribute new functionality and file issues to help in
  :doc:`developing_neon`.


What's Included
---------------

Currently, neon implements and provides examples for the following models
(the name before the colon indicates how the model is referred
to in the source code):

* convnet: Convolutional neural networks
* mlp: Multilayer Perceptrons (Deep neural networks)
* gb: Sparse autoencoders ("Google Brain" style; deprecated)
* autoencoder: Deep autoencoders (reconstructing; Hinton style)
* rnn: Recurrent neural networks
* rbm: Restricted Boltzmann Machines
* dbn: Deep Belief Networks

A key feature of the framework is the ease with which CPU and GPU accelerated
backends can be swapped. In a future release, Flexpoint™ and Nervana HW specific
backends will be added. There are two GPU based backends:
:class:`neon.backends.gpu.GPU` wraps the  NervanaGPU library with fp16
and fp32 Maxwell GPU kernels.
:class:`neon.backends.cc2.GPU` wraps and extends Alex Krizhevsky's
cuda-convnet2 backend.

In addition, the framework provides distributed implementations and examples
using MPI for:

* Convolutional neural networks: data parallel
* Multilayer Perceptrons (Deep neural networks): data and model parallel
* Sparse autoencoders ("Google Brain" style): data parallel

For feature requests and suggestions, email info@nervanasys.com.


Benchmarks
----------

* `Convnet benchmarks <https://github.com/soumith/convnet-benchmarks>`_

Typically we have seen ~20x speedup going from CPU to GPU backends
(Titan GPUs). We are still working on CUDA-aware MPI based multi-GPU
support where we hope to see a roughly linear speedup over single GPU
implementations based for example on work from Alex Krizhevsky. Using MPI
based multi-CPU implementations we are seeing a roughly linear speedup over
single CPU implementations. Once things get more stable we'll add more
detailed benchmarking info to these docs.
