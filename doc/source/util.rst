Utility Functions
=================

Python 2 and 3 Compatibility
----------------------------

.. autosummary::
   :toctree: generated/

   neon.util.compat.PY3

To ensure code runs under python2 and 3, you can utilize the definitions
in :mod:`neon.util.compat`

The :attr:`neon.util.compat.PY3` attribute will be set to True if we are
running under python3 and False otherwise.


Object Instantiation from Config Files
--------------------------------------

.. autosummary::
   :toctree: generated/

   neon.util.factory.Factory

Is the process currently used, but this will eventually be deprecated in favor
of utilizing pyyaml to directly create python objects of a specifie class.
