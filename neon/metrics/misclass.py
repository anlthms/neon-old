# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Misclassification related metrics.
"""

import logging

from neon.metrics.metric import Metric

logger = logging.getLogger(__name__)


class MisclassSum(Metric):
    """
    Metric that counts the number of misclassifications made (prediction does
    not match the reference target exactly).

    See Also: MisclassRate, MisclassPercentage
    """

    def __init__(self):
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
                                             column.
            outputs (neon.backend.Tensor): Predicted outputs.  Must have the
                                           same dimensions as reference.  If
                                           each prediction is a vector, we
                                           treat the inidividual values as
                                           probabilities for that class.
        """
        if reference.shape != outputs.shape:
            raise ValueError("reference dimensions: %s, incompatible with "
                             "outputs dimensions: %s" %
                             (str(reference.shape), str(outputs.shape)))
        self.rec_count += reference.shape[-1]
        if len(outputs.shape) > 1 and outputs.shape[0] > 1:
            # vector of outputs per case.
            self.misclass_sum += (reference.asnumpyarray().argmax(axis=0) !=
                                  outputs.asnumpyarray().argmax(axis=0)).sum()
        else:
            self.misclass_sum += (reference.asnumpyarray().ravel() !=
                                  outputs.asnumpyarray().ravel()).sum()

    def report(self):
        """
        Report the misclassification count.

        Returns:
            int: Misclassification count

        """
        if self.rec_count == 0:
            raise ValueError("No records to count misclassifications on")
        return self.misclass_sum

    def clear(self):
        """
        Reset this metric's calculated value(s)
        """
        self.misclass_sum = 0
        self.rec_count = 0


class MisclassRate(MisclassSum):
    """
    Metric that reports the fraction of misclassifications made (prediction
    does not match the reference target exactly) relative to the total numbe
    of predictions.

    See Also: MisclassSum, MisclassPercentage
    """
    def report(self):
        """
        Report the misclassification rate.

        Returns:
            float: The misclassification rate (will lie between 0.0 and 1.0)
        """
        if self.rec_count == 0:
            raise ValueError("No records to report misclassifications on.")
        else:
            return (self.misclass_sum + 0.0) / self.rec_count


class MisclassPercentage(MisclassRate):

    def report(self):
        """
        Report the misclassification percentage (0-100).

        Returns:
            float: The misclassification percentage (will lie between 0.0 and
                   100.0)
        """
        return super(MisclassPercentage, self).report() * 100.0
