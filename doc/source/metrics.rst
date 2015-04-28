.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Metrics
=======

Metrics are used to quantitatively measure error or some aspect of model
performance (typically against some Dataset partition).

Reporting Metric Values
-----------------------

To report metrics you need to specify them at the Experiment level and choose
FitPredictErrorExperiment as your type.  In the YAML file you'd define the 
metrics dictionary and list the metrics to be computed for each dataset
partition.  Here's an example:

.. code-block:: yaml

    metrics: {
      "train": [
        !obj:metrics.MisclassRate(),
      ],
      "test": [
        !obj:metrics.AUC(),
        !obj:metrics.LogLossMean(),
      ],
      "validation": [
        !obj:metrics.MisclassPercentage(),
      ],
    },


Available Metrics
-----------------

.. autosummary::
   :toctree: generated/

   neon.metrics.misclass.MisclassSum
   neon.metrics.misclass.MisclassRate
   neon.metrics.misclass.MisclassPercentage

   neon.metrics.roc.AUC

   neon.metrics.loss.LogLossSum
   neon.metrics.loss.LogLossMean

   neon.metrics.sqerr.SSE
   neon.metrics.sqerr.MSE
