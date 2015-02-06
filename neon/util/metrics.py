# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

"""
Contains functions for computing error metrics.
"""

def misclass_sum(backend, outputs, targets, preds, labels,
                 misclass, sumval): 
    backend.argmax(outputs, axis=0, out=preds)
    backend.argmax(targets, axis=0, out=labels)
    backend.not_equal(preds, labels, misclass)
    backend.sum(misclass, axes=None, out=sumval)
