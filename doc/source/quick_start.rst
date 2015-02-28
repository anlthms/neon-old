.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Quick start
===========

Currently, the Nervana Framework implements and provides examples for the
following models (the name before the colon indicates how the model is referred
to in the source code):

* convnet: Convolutional neural networks
* mlp: Multilayer Perceptrons (Deep neural networks)
* gb: Sparse autoencoders ("Google Brain" style; no GPU support)
* autoencoder: Deep autoencoders (reconstructing; Hinton style)
* rnn: Recurrent neural networks (forthcoming)
* rbm: Restricted Boltzmann Machines
* dbn: Deep Belief Networks (forthcoming)

A key feature of the framework is the ease with which CPU and GPU accelerated
backends can be swapped. In a future release, Flexpoint™ and Nervana HW
specific backends will be added. The GPU based backend wraps and extends Alex
Krizhevsky's cuda-convnet2 backend.

In addition, the framework provides distributed implementations and examples
using MPI for:

* Convolutional neural networks: data parallel
* Multilayer Perceptrons (Deep neural networks): data and model parallel
* Sparse autoencoders ("Google Brain" style): data parallel

For feature requests and suggestions, email info@nervanasys.com.

Benchmarks
----------

Typically we have seen ~20x speedup going from CPU to GPU backends
(Titan GPUs). We are still working on the CUDA-aware MPI based multi-GPU
support where we hope to see a roughly linear speedup over single GPU
implementations based for example on work from Alex Krizhevsky. Using MPI
based multi-CPU implementations we are seeing a roughly linear speedup over
single CPU implementations. Once things get more stable we'll add more
detailed benchmarking info to these docs.
