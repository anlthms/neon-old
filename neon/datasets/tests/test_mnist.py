# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
import os
from nose.plugins.attrib import attr
import shutil

from neon.datasets.mnist import MNIST
from neon.backends.cpu import CPU
from neon.backends.par import NoPar


class TestMNIST(object):

    tmp_repo = os.path.join(os.path.dirname(__file__), 'repo')

    def setup(self):
        os.makedirs(self.tmp_repo)

    def teardown(self):
        shutil.rmtree(self.tmp_repo, ignore_errors=True)

    @attr('slow')
    def test_get_inputs(self):
        d = MNIST(repo_path=self.tmp_repo)
        d.backend = CPU(rng_seed=0)
        d.backend.init_device()
        d.backend.actual_batch_size = 128
        d.backend.par = NoPar(d.backend)
        inputs = d.get_inputs(train=True)
        # TODO: make this work (numpy import errors at the moment)
        assert inputs['train'] is not None
