# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Sanity check
"""
import argparse
import os
import sys
from neon.util.persist import deserialize


def parse_args():
    parser = argparse.ArgumentParser(description='Run sanity check examples')
    parser.add_argument('--cpu', default=0, help='Run CPU sanity check',
                        type=int)
    parser.add_argument('--gpu', default=0, help='Run GPU sanity check',
                        type=int)
    parser.add_argument('--dist', default=0, help='Run distributed sanity ' +
                        'check', type=int)
    return parser.parse_args()


def sanity_check(conf_file, result):
    dir = os.path.dirname(os.path.realpath(__file__))
    experiment = deserialize(os.path.join(dir, conf_file))
    experiment.run()
    assert experiment.model.result == result

if __name__ == '__main__':
    res = 0
    args = parse_args()
    if args.cpu == 1:
        print('CPU check '),
        sanity_check('check_cpu.yaml', 0.515625)
        print('OK')
    if args.gpu == 1:
        print('GPU check '),
        sanity_check('check_gpu.yaml', 0.5625)
        print('OK')
    if args.dist == 1:
        print('DIST check '),
        sanity_check('check_dist.yaml', 0.4921875)
        print('OK')
    sys.exit(res)
