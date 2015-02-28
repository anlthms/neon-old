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
  that powers our GPU backend.
* [Cython](http://cython.org/) for FlexPoint CPU backend compilation
* [scikit-learn](http://scikit-learn.org) Currently used for AUC performance
  calculations
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
    neon --gpu examples/mlp/mnist-small.yaml

    # For MPI based parallel distributed implementations (single machine):
    # mpirun -n <num_processes> -x <environment_vars> neon -p [-m] <path_to.yaml>
    # ex: 4 process data parallel cnn example from top-level neon dir:
    mpirun -n 4 -x PYTHONPATH bin/neon --datapar \
           examples/convnet/mnist-small.yaml

    # ex: 2 process model parallel cnn example:
    mpirun -n 2 -x PYTHONPATH bin/neon --modelpar \
           examples/convnet/mnist-small.yaml

    # In multi-machine MPI environments need hosts file, data copied to each
    # host, and full paths should be used:
    /<full_path_to_mpirun>/mpirun -n 4 -x LD_LIBRARY_PATH -hostfile hosts \
        /<full_path_to_neon>/neon --datapar \
        /<full_path_to_examples>/convnet/mnist-small.yaml

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
