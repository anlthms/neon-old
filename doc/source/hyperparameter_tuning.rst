.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Hyperparameter optimization
===========================

Finding good hyperparameters for deep networks is quite tedious to do manually
and can be greatly accelerated by performing automated hyperparameter tuning.
To this end, third-party hyperparameter optimization packages can be integrated
with our framework. We currently offer support for Spearmint, available as a fork
at https://github.com/ursk/spearmint/. The package depends on google
protobuf and scipy and uses the flask webserver for visualizing results.

To perform a search over a set of hyperparameters specified in a neon yaml
file, create a new yaml file with the top level experiment of type
:py:class:`neon.experiments.fit_predict_err.FitPredictErrorExperiment`. This
takes an additional argument:

.. code-block:: bash

!obj:experiments.FitPredictErrorExperiment {
  return_item: test,

This ``return_item``, specifies which error
(i.e. for the ``test``, ``training`` or ``validation`` set) should be used as
the objective function for the hyperparameter optimization.

Then in the model specifications of the yaml simply replace a hyper-parameter

.. code-block:: bash

    # specifying a constant learning rate
    learning_rate: 0.1,

with a range over which to search

.. code-block:: bash

    # specifying a range from 0.01 to 0.1 for the learning rate
    learning_rate: !hyperopt lr FLOAT 0.01 0.1,

Where the ``!hyperopt`` flag signals that this is a parameter to be optimized,
followed by a name used to keep track of the parameter, and the type of
variable. Currently, ``FLOAT`` and ``INT`` are supported. The last two
parameters indicate the start and end of the range. An arbitrary number of
parameters can be replaced by ranges. Only scalar, numerical parameters are
supported.

Hyperparameter optimization requires two additional environment variables to
identify the ``spearmint/bin`` directory and the desired location to store
temporary file and results of the experiment, such as:

.. code-block:: bash

    export SPEARMINT_PATH=/path/to/spearmint/spearmint/bin
    export HYPEROPT_PATH=/path/to/hyperopt_experiment

To run a hyperoptimization experiment, call the ``bin/hyperopt`` executable.
To initialize a new experiment, use the ``init`` flag and pass the ``-y``
argument to specify the yaml file containing the hyperparameter ranges, for
example:

.. code-block:: bash

    hyperopt init -y examples/mlp/iris-hyperopt-small.yaml

This creates a spearmint configuration file in protobuf format in the
experiment directory. Then run the experiment by calling with the
``run`` flag and specifying a port with the ``-p`` argument where outputs will
be generated, for example:

.. code-block:: bash

    hyperopt run -p 50000

The output can be viewed in the browser at http://localhost:50000, or by
directly inspecting the files in the experiment directory. The
experiment will keep running indefinitely. It can be interrupted with
``Ctrl+C`` and continued by calling the ``hyperopt run`` command again. To
start a new experiment, reset the previous one first by running:

.. code-block:: bash

    hyperopt reset

Or manually deleting the contents of the experiment directory.
