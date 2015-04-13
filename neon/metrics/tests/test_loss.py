#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rightes reserved.
# ----------------------------------------------------------------------------

import math

from neon.backends.cpu import CPUTensor
from neon.metrics.loss import LogLossSum, LogLossMean


class TestLoss(object):

    def test_logloss_sum(self):
        ll = LogLossSum()
        assert ll.logloss == 0.0
        refs = CPUTensor([[0, 1, 0],
                          [1, 0, 1],
                          [0, 0, 0],
                          [0, 0, 0]])
        preds = CPUTensor([[0.00,    1,    0],
                           [0.09,  0.0, 0.75],
                           [0.01,    0, 0.15],
                           [0.90,    0, 0.10]])
        ll.add(refs, preds)
        assert abs(ll.report() + (math.log(.09) + math.log(1.0 - ll.eps) +
                                  math.log(0.75))) < 1e-6

    def test_logloss_mean(self):
        ll = LogLossMean()
        assert ll.logloss == 0.0
        assert ll.rec_count == 0.0
        refs = CPUTensor([[0, 1, 0],
                          [1, 0, 1],
                          [0, 0, 0],
                          [0, 0, 0]])
        preds = CPUTensor([[0.00,    1,    0],
                           [0.09,  0.0, 0.75],
                           [0.01,    0, 0.15],
                           [0.90,    0, 0.10]])
        ll.add(refs, preds)
        assert abs(ll.report() + (math.log(.09) + math.log(1.0 - ll.eps) +
                                  math.log(0.75)) / 3.0) < 1e-6

#    def test_logloss_alt(self):
#        ll = LogLossMean()
#        refs = CPUTensor([[0, 1, 1, 0],
#                          [1, 0, 0, 1]])
#        preds = CPUTensor([[0.1, 0.9, 0.8, 0.35],
#                           [0.9, 0.1, 0.2, 0.65]])
#        ll.add(refs, preds)
#        assert abs(ll.report() + .21616187468057912) < 1e-6
