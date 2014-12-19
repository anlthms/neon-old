Our in-house developed python machine learning library.

## Installation ##

    # get the latest source
    git clone git@gitlab.localdomain:algorithms/neon.git neon
    cd neon
    
    # build the package, install in your python package path via either:
    make install  # sudo make install
    # or:
    pip install .  # sudo pip install .
    
    # import neon into your app, and you are good to go


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
* [Nervana cuda-convnet2](http://gitlab.localdomain/algorithms/cuda-convnet2/)
  our updated fork of [cuda-convnet2](https://code.google.com/p/cuda-convnet2/)
  that powers our GPU backend.
* [Cython](http://cython.org/) for FlexPoint CPU backend compilation
* [scikit-learn](http://scikit-learn.org) for Google Brain AUC performance
  calculations
* [matplotlib](http://matplotlib.org) for Google Brain feature visualization
* [openmpi](http://www.open-mpi.org), [mpi4py](http://mpi4py.scipy.org) for
  distributed tensors.


## Usage ##

    # neon <path_to.yaml>
    # see the examples directory for sample .yaml files
    neon examples/mnist_numpy_mlp-784-100-10.yaml

    # For MPI based parallel distributed implementations (single machine):
    # mpirun -np <num_processes> -x <environment_vars> neon <path_to.yaml>
    # ex: 4 process cnn example from top-level neon dir:
    mpirun -n 4 -x PYTHONPATH bin/neon \
           examples/mnist_distarray_numpy_cnn-20-50-500-10.yaml

    # In multi-machine MPI environments need hosts file, and full paths should
    # be used:
    /<full_path_to_mpirun>/mpirun -n 4 -x LD_LIBRARY_PATH -hostfile hosts \
        /<full_path_to_neon>/neon \
        /<full_path_to_examples>/mnist_distarray_numpy_cnn-20-50-500-10.yaml

See docs for full details.

## Features ##
* Works with our hardware!  Easy to transition between it and various GPU and
  CPU backends for basic operations
* Highly configurable via yaml files.  Select backend, learning algorithm
  architecture, tuning parameters, and so on.
* Heavily instrumented for performance profiling, debugging, and visualization
* Modular design
* Well documented
* Unit/regression/benchmark timing tested via
  [continuous integration](http://gitlab.localdomain:82)
* python 2 and 3 fully supported (tested against 2.7 and 3.4)


## Issue Tracking ##
* http://nervanasys.atlassian.net/browse/MYL


## Documentation ##
* [Main Source](http://atlas.localdomain:5700)
* [Developer Guide](http://atlas.localdomain:5700/developing_neon.html)
* [API](http://atlas.localdomain:5700/api.html)
* [How to add a model](https://sites.google.com/a/nervanasys.com/wiki/algorithms/neon/how-to-write-a-mylearn-model)
* [Architecture](https://sites.google.com/a/nervanasys.com/wiki/algorithms/neon/architecture)
* [Style and Coding conventions - Google style guide](http://google-styleguide.googlecode.com/svn/trunk/pyguide.html)
   * [Docstring Format - Google style](http://sphinx-doc.org/latest/ext/example_google.html#example-google)


## License ##

Copyright 2014 Nervana Systems Inc.
