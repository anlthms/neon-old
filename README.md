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
* Support for our distributed processor (Nervana Engine™) for deep learning.


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

The Nervana Engine™ supports a set of operations called the [MOP](http://framework.nervanasys.com/docs/latest/ml_operational_layer.html). As long as end user code is written in a MOP compliant manner it will benefit from Nervana Engine's hardware acceleration and scaling abilities. In this way, we have the capability to play with other Deep Learning frameworks such as [theano](https://github.com/Theano/Theano), [torch](https://github.com/torch/torch7) and [caffe](https://github.com/BVLC/caffe). neon models are MOP compliant out of the box. Do not worry if your favorite function is missing! We are still adding MOP functions so please [email us](mailto:framework@nervanasys.com).

## Upcoming libraries

We have separate, upcoming efforts on the following fronts: 

* Distributed models
* Automatic differentiation
* Integration with Nervana Cloud™

## License

We are releasing [neon](https://github.com/NervanaSystems/neon) and [nervanagpu](https://github.com/NervanaSystems/nervanagpu) under an open source [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License. We welcome you to [contact us](mailto:info@nervanasys.com) with your use cases.

