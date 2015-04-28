.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Experiments
===========

Current Implementations
-----------------------

.. autosummary::
   :toctree: generated/

   neon.experiments.experiment.Experiment
   neon.experiments.fit.FitExperiment
   neon.experiments.fit_predict_err.FitPredictErrorExperiment
   neon.experiments.check_grad.GradientChecker
   neon.experiments.generate_output.GenOutputExperiment

.. _extending_experiment:

Adding a new type of Experiment
-------------------------------

#. Subclass :class:`neon.experiments.experiment.Experiment`

.. _gen_predictions:

Saving Results
--------------
Model predictions can be saved to disk when running a
:class:`neon.experiments.fit_predict_err.FitPredictErrorExperiment`.  To do so
add the following inside your top-level experiment:

.. code-block:: yaml

    inference_sets: ['train', 'test'],

This will result in the generation of two new python serialized object (.pkl)
files being written to the directory in which your dataset resides.  The first
will contain model outputs from running the specified dataset through the
trained model.  The second file will contain the expected target values for the
same set of data.

In the example above we've requested saved outputs for the training and test
datasets, though 'validation' datasets can also be included if supported.
