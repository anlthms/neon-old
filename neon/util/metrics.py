# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

"""
Contains functions for computing error metrics.
"""


def misclass_sum(backend, targets, outputs, predlabels, labels,
                 misclass, retval):
    backend.argmax(targets, axis=0, out=labels)
    backend.argmax(outputs, axis=0, out=predlabels)
    backend.not_equal(predlabels, labels, misclass)
    backend.sum(misclass, axes=None, out=retval)


def auc(backend, targets, outputs):
    from sklearn import metrics
    return metrics.roc_auc_score(targets.asnumpyarray().ravel(),
                                 outputs.asnumpyarray().ravel())


def logloss(backend, targets, outputs, sums, temp, retval, eps=1e-15):
    backend.clip(outputs, eps, 1.0 - eps, out=outputs)
    backend.sum(outputs, axes=0, out=sums)
    # XXX: work around lack of broadcasting in gpu backend.
    for row in range(temp.shape[0]):
        temp[row] = sums
    backend.divide(outputs, temp, out=temp)
    backend.log(temp, out=temp)
    backend.multiply(targets, temp, out=temp)
    backend.sum(temp, axes=None, out=retval)
