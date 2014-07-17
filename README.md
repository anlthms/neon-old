Our in-house developed python machine learning library.

## Installation ##

    # get the latest source
    git clone git@192.168.20.2:algorithms/mylearn.git mylearn
    cd mylearn
    
    # build the package, install in your python package path via either:
    make install
    # or:
    pip install .
    
    # import mylearn into your app, and you are good to go


### Required Dependencies ###
We want to strive to have as few of these as possible
* python 2.7 or higher

### Optional Dependencies ###
* nose (for running tests)
* sphinx (for documentation generation)
* numpy (for certain backends)


## Usage ##

TODO: add a quick vignette/tutorial/walkthrough training a basic neural net


## Features ##
* Works with our hardware!  Easy to transition between it and various GPU and
  CPU backends for basic operations
* Highly configurable via yaml files.  Select backend, learning algorithm
  architecture, tuning parameters, and so on.
* Heavily instrumented for performance profiling, debugging, and visualization
* Modular design
* Well documented
* Unit/regression/benchmark timing tested
* python 2 and 3 support. TODO: utilize 2to3.py?


## Issue Tracking ##
* http://192.168.20.2/algorithms/mylearn/issues


## Documentation ##
* [API](http://192.168.20.3:8000) TODO: update link to a more permanent
  host/path
* [How to add a model](https://sites.google.com/a/nervanasys.com/wiki/algorithms/mylearn/how-to-write-a-mylearn-model)
* [Architecture](https://sites.google.com/a/nervanasys.com/wiki/algorithms/mylearn/architecture)
* [Style and Coding conventions - google style guide](http://google-styleguide.googlecode.com/svn/trunk/pyguide.html) TODO: integrate flake8, google style guide?
