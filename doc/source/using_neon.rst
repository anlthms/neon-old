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
Neon provides distributed implementations of convnets and sparse autoencoders in addition to the non-distributed implementations.
It has been tested with `OpenMPI 1.8.1 <http://www.open-mpi.org/software/ompi/v1.8/>`_ and `mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_.

1. Install OpenMPI:

.. code-block:: bash

    cd <openmpi_source_dir>
    ./configure --prefix=/<path_to_install_openmpi> --with-cuda
    make all
    sudo make install

Make sure that PATH includes /<path_to_openmpi>/bin and LD_LIBRARY_PATH includes /<path_to_openmpi>/lib

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

In distributed environments with multiple nodes full paths might be needed for mpirun and neon, for e.g.:

.. code-block:: bash

    /<full_path_to_mpirun>/mpirun -np 4 -x LD_LIBRARY_PATH -hostfile hosts 
        /<full_path_to_neon>/neon 
        /<full_path_to_examples>/mnist_distarray_cpu_cnn-20-50-500-10.yaml

LD_LIBRARY_PATH should point to /<path_to_openmpi>/lib. A common file system is assumed.
