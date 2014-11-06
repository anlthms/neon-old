"""
Sanity check
"""
import os
import neon
import logging
from neon.util.persist import deserialize


def sanity_check(conf_file, result):
    dir = os.path.dirname(os.path.realpath(__file__))
    experiment = deserialize(os.path.join(dir, conf_file))
    experiment.run()
    assert experiment.model.result == result

if __name__ == '__main__':
    sanity_check('check_cpu.yaml', 0.3203125)
    sanity_check('check_gpu.yaml', 0.390625)
    print 'OK'
