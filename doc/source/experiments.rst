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
   neon.experiments.write_error_to_file.WriteErrorToFile

.. _extending_experiment:

Adding a new type of Experiment
-------------------------------

#. Subclass :class:`neon.experiments.experiment.Experiment`

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

Performance Metrics
-------------------
As a model trains, the current training error is reported after each epoch.  To
specify additional metrics to report at the end of training, indicate what
datasets to report on, along with the specific metrics.  Inside your top-level
experiment, add the following to your yaml file:

.. code-block:: yaml

    inference_sets: ['test'],
    inference_metrics: ['auc', 'misclass rate']

In the example above we asked for 'auc' (area under the ROC curve), and
misclassification rate to be reported.  The currently implemented set of
metrics includes:

* auc - Area under the ROC curve
* misclass rate - Misclassification rate.  The proportion of exemplars whose
  label value differed from the predicted value output by the model.
* log loss - The negative log likelihood of the true target labels given
  predicted output probabilities from the model.
