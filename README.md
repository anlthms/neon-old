# Public beta release of neon

We are pleased to announce the public beta release of [neon](https://github.com/NervanaSystems/neon), which is <span style="color:red;">NE</span>rvana's pyth<span style="color:red;">ON</span> based Deep Learning Framework! We have designed it with the following functionality in mind:

* YAML for easy model specification
* Python for easy model hacking and support for many data formats
* Support for commonly used models: convnets, MLPs, RNNs, LSTMs, autoencoders, RBMs
* Support for common learning rules, activation functions and cost functions
* Comparing performance of alternate numeric representations with fp32 for Deep Learning
* Swappable hardware backends: write code once and then deploy on CPUs, GPUs, or Nervana hardware

Features that are unique to neon include:

* Tight integration with [nervanagpu](https://github.com/NervanaSystems/nervanagpu) kernels for fp16 and fp32 ([benchmarks](https://github.com/soumith/convnet-benchmarks)) on Maxwell GPUs. These are the fastest implementations of the benchmark deep networks.
* 4.3s/macrobatch on AlexNet on Titan X (Full run on 1 GPU ~ 45 hrs)
* Out of the box [fp16 AlexNet model](https://github.com/NervanaSystems/neon/tree/master/examples/convnet/i1k-alexnet-fp16.yaml) that has the same accuracy as [fp32](https://github.com/NervanaSystems/neon/tree/master/examples/convnet/i1k-alexnet-fp32.yaml)
* Integration with our fork ([cudanet](https://github.com/NervanaSystems/cuda-convnet2)) of Alex Krizhevsky's cuda-convnet2 library for Kepler GPU support
* Support for our distributed processor (Nervana Engine&trade;) for deep learning.


We use neon internally at Nervana to solve our customers' problems across many [domains](http://www.nervanasys.com/products/). We are hiring across several roles. Apply [here](http://www.nervanasys.com/careers/)!

## Getting started

Basic information to get started is below. Please consult the [full documentation](http://framework.nervanasys.com/docs/latest) for more information.

### Installation

* [Local install and dependencies](http://framework.nervanasys.com/docs/latest/using_framework.html#installation)
* Cloud-based access ([email us](mailto:demo@nervanasys.com) for an account)

There are several examples built-in to neon in the `examples` directory for a user to get started. The YAML format is plain-text and can be edited to change various aspects of the model. See the `ANNOTATED_EXAMPLE.yaml` for some of the definitions and possible choices.

### Running a simple MNIST model

	neon examples/mlp/mnist-small.yaml
	
### Running an Alexnet model

In [fp32](https://github.com/NervanaSystems/neon/tree/master/examples/convnet/i1k-alexnet-fp32.yaml):

	# for nervangpu
	neon -g nervanagpu examples/convnet/i1k-alexnet-fp32.yaml
	
	# for cudanet
	neon -g cudanet examples/convnet/i1k-alexnet-fp32.yaml
	
`-g` stands for GPU hardware backend.

In [fp16](https://github.com/NervanaSystems/neon/tree/master/examples/convnet/i1k-alexnet-fp16.yaml):

	neon -g nervanagpu examples/convnet/i1k-alexnet-fp16.yaml

### Code organization

	backends    --- implementation of different hardware backends
	datasets    --- support for common datasets CIFAR-10, ImageNet, MNIST etc.
	diagnostics --- hooks to measure timing and numeric ranges
	hyperopt    --- hooks for hyperparameter optimization
	layers      --- layer code
	models      --- model code
	optimizers  --- learning rules
	transforms  --- activation & cost functions
	metrics     --- performance evaluation metrics
  

### Documentation

The complete documentation for neon is available [here](http://framework.nervanasys.com/docs/latest) 

### Issues

For any bugs or feature requests please create a ticket [here](https://github.com/NervanaSystems/neon/issues).

## Machine learning OPerations (MOP) Layer

The Nervana Engine&trade; supports a set of operations called the [MOP](http://framework.nervanasys.com/docs/latest/ml_operational_layer.html). As long as end user code is written in a MOP compliant manner it will benefit from Nervana Engine's hardware acceleration and scaling abilities. In this way, we have the capability to play with other Deep Learning frameworks such as [theano](https://github.com/Theano/Theano), [torch](https://github.com/torch/torch7) and [caffe](https://github.com/BVLC/caffe). neon models are MOP compliant out of the box. Do not worry if your favorite function is missing! We are still adding MOP functions so please [email us](mailto:framework@nervanasys.com).

## Upcoming libraries

We have separate, upcoming efforts on the following fronts: 

* Distributed models
* Automatic differentiation
* Integration with Nervana Cloud&trade;

## License

We are releasing [neon](https://github.com/NervanaSystems/neon) and [nervanagpu](https://github.com/NervanaSystems/nervanagpu) under an open source [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License. We welcome you to [contact us](mailto:info@nervanasys.com) with your use cases.

=======
Our in-house developed python machine learning library.

## Installation ##

    # get the latest source
    git clone https://github.com/NervanaSystems/neon.git
    cd neon

    # build the package, install in your python package path via either:
    make install  # sudo make install
    # or:
    pip install .  # sudo pip install .

    # run the included command line executable to start launching your
    # experiments
    neon --help


### Required Dependencies ###
We strive to have as few of these as possible
* python 2.7 or higher
* [numpy](http://www.numpy.org/) for certain backends and dataset parsing
* [pyyaml](http://pyyaml.org/) for config file parsing

### Optional Dependencies ###
These provide additional functionality, and assist developers
* [nose](https://nose.readthedocs.org/en/latest/) for running tests
* [sphinx](http://sphinx-doc.org/) for documentation generation
  * sphinxcontrib-napoleon for google style autodoc parsing
* [flake8](https://flake8.readthedocs.org/) for style checking
  * [pep8-naming](https://pypi.python.org/pypi/pep8-naming) plugin for variable
    name checking
* [Nervana cuda-convnet2](http://github.com/NervanaSystems/cuda-convnet2/)
  our updated fork of [cuda-convnet2](https://code.google.com/p/cuda-convnet2/)
  that powers the cudanet GPU backend.
* [nervanagpu](http://github.com/NervanaSystems/nervanagpu/) our in-house
  developed fp16/fp32 Maxwell GPU backend.
* [pycuda](http://mathema.tician.de/software/pycuda/) required for our
  nervanagpu backend
* [Cython](http://cython.org/) for FlexPoint CPU backend compilation
* [matplotlib](http://matplotlib.org) for RNN feature visualization
* [openmpi](http://www.open-mpi.org), [mpi4py](http://mpi4py.scipy.org) for
  distributed tensors.


## Usage ##

    # neon <path_to.yaml>
    # see the examples directory for sample .yaml files
    neon examples/mlp/mnist-small.yaml

    # to see the list of options controlling execution type:
    neon --help

    # for GPU based runs, you need to have a CUDA capable GPU card installed
    # then run:
    neon --gpu cudanet examples/mlp/mnist-small.yaml
    # to run on the cuda-convnet2 backend (fp32 precision, supports Kepler) or
    neon --gpu nervanagpu examples/mlp/mnist-small.yaml
    # to run on the nervanagpu backend (supports fp16 and fp32)

    # For MPI based parallel distributed implementations (single machine):
    # mpirun -n <num_processes> [-x <environment_vars>] neon -p [-m] <path_to.yaml>
    # ex: 4 process data parallel cnn example:
    mpirun -n 4 neon --datapar examples/convnet/mnist-small.yaml

    # ex: 2 process model parallel cnn example:
    mpirun -n 2 neon --modelpar examples/convnet/mnist-small.yaml

See docs for full details.

## Features ##
* Works with our hardware!  Easy to transition between it and various GPU and
  CPU backends for basic operations
* Highly configurable via yaml files.  Select learning algorithm, architecture,
  tuning parameters, and so on.
* Heavily instrumented for performance profiling, debugging, and visualization
* Modular design
* Well documented
* Unit/regression/benchmark timing tested via
  [continuous integration](http://gitlab.localdomain:82)
* python 2 and 3 fully supported (tested against 2.7 and 3.4)


## Issue Tracking ##
* https://github.com/NervanaSystems/neon/issues
* [internal tracking](http://nervanasys.atlassian.net/browse/MYL)


## Documentation ##
* [Main Source](http://framework.nervanasys.com/docs/latest)
* [Developer Guide](http://framework.nervanasys.com/docs/latest/developing_framework.html)
* [API](http://framework.nervanasys.com/docs/latest/api.html)
* [Extending the Framework](http://framework.nervanasys.com/docs/latest/developing_framework.html#extending-the-framework)
* [Architecture](https://framework.nervanasys.com/docs/latest/developing_framework.html#architecture)
* [Style and Coding conventions - Google style guide](http://google-styleguide.googlecode.com/svn/trunk/pyguide.html)
   * [Docstring Format - Google style](http://sphinx-doc.org/latest/ext/example_google.html#example-google)


## License ##

Please see LICENSE file for complete details.
