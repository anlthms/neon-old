from nose.tools import assert_raises

from neon.datasets.dataset import Dataset


class TestDataset(object):

    def test_load(self):
        d = Dataset()
        assert_raises(NotImplementedError, d.load)