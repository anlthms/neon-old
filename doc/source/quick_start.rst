Quick start
===========

Currently, neon implements and provides examples for the following models:

* Convolutional neural networks
* Multilayer Perceptrons (Deep neural networks)
* Restricted Boltzmann Machines
* Sparse autoencoders (Google Brain style)
* Deep autoencoders (reconstructing; Hinton style)
* Recurrent neural networks (forthcoming)

In addition, neon provides distributed implementations and examples for:

* Convolutional neural networks
* Multilayer Perceptrons (Deep neural networks)
* Sparse autoencoders (Google Brain style)

Typically we have seen ~20x speedup going from CPU to GPU (Titan GPUs). We are still working on the CUDA-aware MPI based multi-GPU support where we hope to see a roughly linear speedup over single GPU implementations based for example on work from Alex Krizhevsky. Using MPI based multi-CPU implementations we are seeing a roughly linear speedup over single CPU implementations. Once things get more stable we'll add more detailed benchmarking info to the docs.