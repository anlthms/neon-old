# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains various loss related metrics ex. log loss
"""

import numpy

from neon.metrics.metric import Metric


class LogLossSum(Metric):
    """
    Logistic loss (aka cross-entropy loss) for a multi-class classification
    task.  Defined to be the negative log of the likelihood summed across all
    data points received.

    Arguments:
        eps (float, optional): Amount to clip values by to prevent potential
                               numeric difficulties (taking log of 0).

    See Also:
        LogLossMean

    References:
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


class LogLossMean(LogLossSum):
    """
    Logistic loss (aka cross-entropy loss) for a multi-class classification
    task.  Defined to be the negative log of the likelihood averaged across all
    data points received.

    Arguments:
        eps (float, optional): Amount to clip values by to prevent potential
                               numeric difficulties (taking log of 0).

    See Also:
        LogLossSum
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
        super(LogLossMean, self).add(reference, outputs)
        self.rec_count += reference.shape[-1]

    def report(self):
        """
        Report the mean log loss value

        Returns:
            float: log loss mean value
        """
        return super(LogLossMean, self).report() / self.rec_count

    def clear(self):
        """
        Reset this metric's calculated value
        """
        super(LogLossMean, self).clear()
        self.rec_count = 0.0
