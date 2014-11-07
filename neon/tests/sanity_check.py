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
    sanity_check('check_cpu.yaml', 0.515625)
    sanity_check('check_gpu.yaml', 0.4921875)
    sanity_check('check_dist.yaml', 0.5078125)
    print 'OK'
