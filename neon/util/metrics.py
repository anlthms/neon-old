# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

"""
Contains functions for computing error metrics.
"""


def misclass_sum(backend, reference, outputs, predlabels, labels, misclass,
                 retval, topk=1):
    if reference.shape[0] == 1:
        labels[:] = reference
    else:
        backend.argmax(reference, axis=0, out=labels)
    backend.argmax(outputs, axis=0, out=predlabels)
    backend.not_equal(predlabels, labels, misclass)
    backend.sum(misclass, axes=None, out=retval)


def auc(backend, reference, outputs):
    from sklearn import metrics
    return metrics.roc_auc_score(reference.asnumpyarray().ravel(),
                                 outputs.asnumpyarray().ravel())


def logloss(backend, reference, outputs, sums, temp, retval, eps=1e-15):
    backend.clip(outputs, eps, 1.0 - eps, out=outputs)
    backend.sum(outputs, axes=0, out=sums)
    backend.divide(outputs, sums, out=temp)
    backend.log(temp, out=temp)
    backend.multiply(reference, temp, out=temp)
    backend.sum(temp, axes=None, out=retval)


def logloss_and_misclass(backend, reference, outputs, labellogprob, top1error,
                         topkerror, topk, sums, eps=1e-15):
    backend.clip(outputs, eps, 1.0 - eps, out=outputs)
    backend.sum(outputs, axes=0, out=sums)
    backend.divide(outputs, sums, outputs)
    backend.logloss_and_misclass(reference, outputs, labellogprob, top1error,
                                 topkerror, topk)
