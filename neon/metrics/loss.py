# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains various loss related metrics ex. log loss
"""

import numpy

from neon.metrics.metric import Metric


class LogLoss(Metric):
    """
    Logistic loss (aka cross-entropy loss) for a multi-class classification
    task.  Defined to be the negative log of the likelihood

    Arguments:
        eps (float, optional): Amount to clip values by to prevent potential
                               numeric difficulties (taking log of 0).

    See Also:
        Bishop2006 (p. 209)
    """
    def __init__(self, eps=1e-15):
        self.eps = eps
        self.clear()

    def add(self, reference, outputs):
        """
        Add the the expected reference and predicted outputs passed to the set
        of values used to calculate this metric.

        Arguments:
            reference (neon.backend.Tensor): Ground truth, expected outcomes.
                                             If each outcome is a vector, we
                                             expect it to be a column vector,
                                             with each case in a separate
                                             (one-hot encoded) column.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.  To
                                           prevent numeric difficulties, output
                                           probabilities will be scaled to lie
                                           within [self.eps, 1 - self.eps]
        """
        if reference.shape != outputs.shape:
            raise ValueError("reference dimensions: %s, incompatible with "
                             "outputs dimensions: %s" %
                             (str(reference.shape), str(outputs.shape)))
        # clip and normalize predictions
        preds = outputs.asnumpyarray().clip(self.eps, (1.0 - self.eps))
        preds = numpy.log(preds / preds.sum(axis=0))
        self.logloss += (reference.asnumpyarray() * preds).sum()

    def report(self):
        """
        Report the log loss value

        Returns:
            float: log loss value
        """
        return - self.logloss

    def clear(self):
        """
        Reset this metric's calculated value
        """
        self.logloss = 0.0
