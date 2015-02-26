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

Adding a new type of experiment
--------------------------------

* Subclass :class:`neon.experiments.experiment.Experiment`
