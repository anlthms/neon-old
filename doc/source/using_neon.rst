.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Using Neon
==========

Installation
------------
.. code-block:: bash

    # get the latest source
    git clone git@gitlab.localdomain:algorithms/neon.git neon
    cd neon

    # configure optional backends like GPU, distributed processing by editing
    # setup.cfg (set items to 1 to enable):
    vi setup.cfg

    # to install system wide:
    make install  # sudo make install on Linux

    # or to build for working locally in the source tree
    # (useful for active development)
    make develop  # sudo make develop on Linux
    # or
    make build  # will require updating PYTHONPATH to point at neon dir

Running
-------
A command line executable named neon is included:

.. code-block:: bash

    neon my_example_cfg.yaml

Some example yaml files are available in the ``examples`` directory of the
package.  To understand options available with the command you can issue
``-h`` or ``--help``:

.. code-block:: bash

    neon --help

If working locally, you'll need to make sure the root ``neon`` directory is in
your python path:

.. code-block:: bash

    PYTHONPATH="." bin/neon my_example_cfg.yaml

Configuration Setup
-------------------
Initial build type and required dependency handling can be controlled either by
editing the ``setup.cfg`` file prior to installation, or by passing arguments
to the ``make`` command.  Below is an example showing the default values for
``setup.cfg``:

.. highlight:: ini

.. literalinclude:: ../../setup.cfg
   :linenos:

As shown, the default set of options is fairly restrictive, so only the CPU
based backend will be available.  If you have a CUDA capable GPU, you'll
likely want to set ``GPU=1``.  If you plan to run unit tests, build
documentation or develop neon, you'll want to set ``DEV=1``.  If you would
like to run your model training in parallel via MPI you'll need to first set
``DIST=1``.

To override what is defined in ``setup.cfg``, one can pass the appropriate
options on the command-line (useful when doing in place development).  Here's
an example:

.. code-block:: bash

    make -e GPU=1 DEV=1 test

Experiment File Format
----------------------
A `YAML <http://www.yaml.org/>`_ configuration file is used to control the
design of each experiment.  Below is a fully annotated example showing the
process to train and run inference on a toy network:

.. highlight:: bash

.. literalinclude:: ../../examples/ANNOTATED_EXAMPLE.yaml
   :linenos:

Installing MPI on an Ubuntu cluster (for distributed models)
------------------------------------------------------------
Neon provides distributed implementations of convnets and sparse autoencoders
in addition to the non-distributed implementations.
It has been tested with
`OpenMPI 1.8.1 <http://www.open-mpi.org/software/ompi/v1.8/>`_ and
`mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_.

1. Install OpenMPI:

.. code-block:: bash

    cd <openmpi_source_dir>
    ./configure --prefix=/<path_to_install_openmpi> --with-cuda
    make all
    sudo make install

Make sure that PATH includes /<path_to_openmpi>/bin and LD_LIBRARY_PATH
includes /<path_to_openmpi>/lib

2. Install mpi4py:

.. code-block:: bash

  # set DIST=1 in setup.cfg then run:
  make install
  # or
  make -e DIST=1 install
  # or
  cd <mpi4py_source_dir>
	sudo python setup.py build --configure install

3. Setup /etc/hosts with IPs of the nodes.
e.g.:

.. code-block:: bash

	192.168.1.1 titan
	192.168.1.2 wimp

4. Setup a hosts file to use with MPI -hostfile option.
For additional info refer `here <http://cs.calvin.edu/curriculum/cs/374/homework/MPI/01/multicoreHostFiles.html>`_.
e.g.:

.. code-block:: bash

	titan slots=2
	wimp slots=2

Running MPI models
------------------
For MPI based distributed implementations, on a single node:

.. code-block:: bash

    # mpirun -np <number_of_processes> -x <env_vars_to_export> neon examples/<path_to.yaml>
    # where PYTHONPATH includes ./
    mpirun -np 4 -x PYTHONPATH bin/neon examples/mnist_distarray_cpu_cnn-20-50-500-10.yaml

In distributed environments with multiple nodes full paths might be needed
for mpirun and neon, for e.g.:

.. code-block:: bash

    /<full_path_to_mpirun>/mpirun -np 4 -x LD_LIBRARY_PATH -hostfile hosts
        /<full_path_to_neon>/neon
        /<full_path_to_examples>/mnist_distarray_cpu_cnn-20-50-500-10.yaml

LD_LIBRARY_PATH should point to /<path_to_openmpi>/lib. A common file system
is assumed.


Serializing models
------------------
Serializing / deserialization is supported in a general way for all objects that
appear in the yaml file. To serialize an object (i.e. to create a checkpoint),
add a line in the yaml file like
.. code-block:: bash

  model: !obj:neon.models.mlp.MLPL {
    serialized_path: './serialized_file_path.pkl',

and the model !obj will be serialized into a pickel file at the specified path.
This is possible for any `!obj` object, e.g. individual layers. To warm-start
from a checkpoint, use a yaml file that contains
.. code-block:: bash

  model: !obj:neon.models.mlp.MLPL {
    deserialized_path: './serialized_file_path.pkl',
    # overwrite_list: [layers],

and everything inside the model (layers, batch_size, backend) will be taken from
the pickle file. The optional parameter `overwrite_list` specifies a list of
objects that should not be taken from the pickel object, but from the yaml file.


Object Localization
-------------------
(NOT SURE THIS BELONGS UNDER using_neon, MOVE SOMEWHERE ELSE?)
Obeject localization is currently supported as a two-stage process, where first
a network is trained on small, cropped patches of an object, and the feature
detectors learned by this network are then reused to perform inference on a
set of possibly much larger test images. This requires using a pair of yaml files,
one for training that specifies a crop dataset and a data layer with the correct
`ofmshape` size, that serializes the layer stack. The second yaml file contains
the full images (TODO support to set corresponding `ofmshape` size). An example
of this is given for the Hurrican dataset in the files
`hurricane_cpu_cnn_multivar.yaml` and `hurricane_cpu_cnn_multivar_loca.yaml`
in the `neon/tests` directory.


Hyperparameter optimization
---------------------------
Finding good hyperparameters for deep networks is quite tedious to do manually
and can be greatly accelerated by performing automated hyperparameter tuning.
To this end, third-party hyperparameter optimization packages can be integrated
with neon. We currently offer support for Spearmint, available as a fork
at `https://github.com/ursk/spearmint/`. The package depends on google
protobuf and uses the flask webserver for visualizing results.

To perform a serach over a set of hyperparameters specified in a neon yaml
file, create a new yaml file with the top level experiment type
`neon.experiments.write_error_to_file.WriteErrorToFile`. This takes two
additional arguments:

.. code-block:: bash

    !obj:neon.experiments.write_error_to_file.WriteErrorToFile {
      filename: neon_result_validation.txt,
      item: test,

The first, `filename` specifies the name of the file the result of the run
should be written to, and the second, `item`, specifies which error
(i.e. for the `test`, `training` or `validation` set) should be used as the
objective function for the hyperparameter optimization.

Then in the model specifications of the yaml simply replace a hyper-parameter

.. code-block:: bash

    # specifying a constant learning rate
    learning_rate: 0.1,

with a range over which to search

.. code-block:: bash

    # specifying a range from 0.01 to 0.1 for the learning rate
    learning_rate: !hyperopt lr FLOAT 0.01 0.1,

where the !hyperopt flag signals that this is a parameter to be optimized,
followed by a name used to keep track of the parameter, and the type of
variable. Currently, FLOAT and INT are supported. The last two parameters
indicate the start and end of the range. An arbitrary number of parameters
can be replaced by ranges. Only scalar, numerical parameters are supported.

To run a hyperoptimization experiment, call the `bin/hyperopt` executable.
To initialize a new exeriment, use the `init` flag and pass the `-y` argument
to specify the yaml file containing the hyperparameter ranges, for example

.. code-block:: bash

    PYTHONPATH='`pwd`' bin/hyperopt init -y examples/hyper_iris_small.yaml

this creates a speramint configuration file in proptobuf format in the
`neon/hyperopt/expt` directory. Then run the experiment by calling with the
`run` flag and specifying a port with the `-p` argument where outputs will be
generated, for example

.. code-block:: bash

    PYTHONPATH='`pwd`' bin/hyperopt run -p 50000

The output can be viewed in the browser at `http://localhost:50000`, or by
directly inspecting the files in the `neon/hyperopt/expt` directory. The
experiment will keep running indefinitely. It can be interrupted with `Ctrl+C`
and continued by calling the `hyperopt run` command again. To start a new
experiment, reset the previous one first by running

.. code-block:: bash

    PYTHONPATH='`pwd`' bin/hyperopt reset

or manually deleting the contents of the `neon/hyperopt/expt` directory.
