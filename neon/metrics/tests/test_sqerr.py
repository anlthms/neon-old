#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rightes reserved.
# ----------------------------------------------------------------------------

from neon.backends.cpu import CPUTensor
from neon.metrics.sqerr import SSE, MSE


class TestSquaredError(object):

    def test_sse(self):
        ss = SSE()
        assert ss.value == 0.0
        refs = CPUTensor([0.00,  1, 0.7, -0.4])
        preds = CPUTensor([0.2, -1, 0.5,  2.4])
        ss.add(refs, preds)
        assert abs(ss.report() - (0.2**2 + 2**2 + 0.2**2 + 2.8**2)) < 1e-5

    def test_sse_mat(self):
        ss = SSE()
        assert ss.value == 0.0
        refs = CPUTensor([[0, 1, 0.7],
                          [0.5, -3, 0]])
        preds = CPUTensor([[0.2, 1, 0.5],
                           [0, -5.5, 0.10]])
        ss.add(refs, preds)
        assert abs(ss.report() - (0.2**2 + 0**2 + 0.2**2 +
                                  0.5**2 + 2.5**2 + .10**2)) < 1e-5

    def test_mse(self):
        ms = MSE()
        assert ms.value == 0.0
        refs = CPUTensor([0.00,  1, 0.7, -0.4])
        preds = CPUTensor([0.2, -1, 0.5,  2.4])
        ms.add(refs, preds)
        assert abs(ms.report() -
                   ((0.2**2 + 2**2 + 0.2**2 + 2.8**2) / 4.0)) < 1e-5

    def test_mse_mat(self):
        ms = MSE()
        assert ms.value == 0.0
        assert ms.rec_count == 0.0
        refs = CPUTensor([[0, 1, 0.7],
                          [0.5, -3, 0]])
        preds = CPUTensor([[0.2, 1, 0.5],
                           [0, -5.5, 0.10]])
        ms.add(refs, preds)
        assert abs(ms.report() - ((0.2**2 + 0**2 + 0.2**2 +
                                   0.5**2 + 2.5**2 + .10**2) / 6.0)) < 1e-6
