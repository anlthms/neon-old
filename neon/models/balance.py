# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains code to train Google Brain models and run inference.
"""

import logging
import math
import os

from neon.models.layer import BranchLayer
from neon.models.mlp import MLP
from neon.util.persist import ensure_dirs_exist
from neon.transforms.sum_squared import SumSquaredDiffs
from neon.transforms.xcov import XCovariance

import numpy as np
import time

logger = logging.getLogger(__name__)


class Balance(MLP):

    def __init__(self, **kwargs):
        super(Balance, self).__init__(**kwargs)
        self.bs = self.backend.wrap(self.batch_size)
        self.link_layers()

    def link_layers(self):
        """
        Run through the layers and create a linked list structure
        """
        self.ldict = {}
        # first map all the layers to their names
        for l in self.layers:
            if l.name in self.ldict:
                raise ValueError("layer name %s already used" % l.name)
            self.ldict[l.name] = l
            l.prevs = []
            l.nexts = []

        # now start linking them up
        for l in self.layers:
            if len(l.prev_names) == 0:
                continue
            try:
                l.prevs = map(lambda x: self.ldict[x], l.prev_names)
            except KeyError:
                raise KeyError("Layer name not found: %s" % l.prev_names)
            for p in l.prevs:
                isinstance(l, BranchLayer)
                p.nexts = [l]

        # check for orphans and set the input and terminal nodes
        for l in self.layers:
            if len(l.prevs) == 0 and len(l.nexts) == 0:
                print "Orphan found: %s" % l.name
                continue
            if len(l.prevs) == 0:
                self.input_layer = l
            if len(l.nexts) == 0:
                self.output_layer = l

        for l in self.layers:
            print l.name
            for lp in l.prevs:
                print '\t p ', lp.name
            for ln in l.nexts:
                print '\t n ', ln.name


    def get_error(self, targets, inputs):
        error = 0.0
        # error += self.cost[1].apply_function(targets)
        for c,t in zip(self.cost, [inputs, targets, targets]):
            error += c.apply_function(t)
        return error

    def fprop(self, inputs):
        vqueue = [self.input_layer]
        y = inputs
        while len(vqueue) != 0:
            l = vqueue.pop(0)
            l.fprop(y)
            y = l.output
            vqueue.extend(l.nexts)

    def bprop(self, targets, inputs):
        cost_inputs = [inputs, targets, targets]
        cost_errors = map(lambda x: x[0].apply_derivative(x[1]),
                           zip(self.cost, cost_inputs))

        for c, err in zip(self.cost, cost_errors):
            self.backend.divide(err, self.bs, out=err)
            berror = err
            vqueue = [c.olayer]
            while len(vqueue) != 0:
                l = vqueue.pop(0)
                if len(l.prevs) > 0:
                    l.bprop(berror, l.prevs[0].output)
                    berror = l.berror
                    vqueue.extend(l.prevs)
                else:
                    l.bprop(berror, inputs)

    def get_classifier_output(self):
        return self.ldict['classlayer'].output
