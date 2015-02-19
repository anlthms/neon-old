# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

import os

from neon.util.persist import ensure_dirs_exist


class TestSerialize(object):

    def test_dir_creation(self):
        test_dir = os.path.join('.', 'temp_dir')
        test_file = os.path.join(test_dir, 'temp_file.txt')
        assert not os.path.exists(test_file)
        assert not os.path.isdir(test_dir)
        ensure_dirs_exist(test_file)
        try:
            assert os.path.isdir(test_dir)
        finally:
            try:
                os.rmdir(test_dir)
            except OSError:
                pass

    def test_empty_dir_path(self):
        test_file = ('temp_file.txt')
        assert not os.path.exists(test_file)
        assert not os.path.isdir(test_file)
        ensure_dirs_exist(test_file)
        try:
            assert not os.path.isdir(test_file)
            assert not os.path.exists(test_file)
        finally:
            try:
                os.rmdir(test_file)
            except OSError:
                pass
