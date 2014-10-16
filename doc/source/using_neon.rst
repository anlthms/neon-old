Using Neon
==========

Installation
------------
.. code-block:: bash

    # get the latest source
    git clone git@gitlab.localdomain:algorithms/neon.git neon
    cd neon

    # to install system wide:
    make install  # sudo make install on Linux
    # or:
    pip install .  # sudo pip install . on Linux

    # or to build for working locally in the source tree
    # (useful for active development)
    make build

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

Configuration File Format
-------------------------
A `YAML <http://www.yaml.org/>`_ configuration file is used to control the
design of each experiment.  Below is a fully annotated example showing the
process to train and run inference on a toy network:

.. highlight:: bash

.. literalinclude:: ../../examples/ANNOTATED_EXAMPLE.yaml
   :linenos:
