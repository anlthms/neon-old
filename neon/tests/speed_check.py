# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Speed check
"""
import argparse
import os
import sys
import time
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


def speed_check(conf_file, num_epochs):
    dir = os.path.dirname(os.path.realpath(__file__))
    experiment = deserialize(os.path.join(dir, conf_file))
    experiment.model.num_epochs = num_epochs
    start = time.time()
    experiment.run()
    return (time.time() - start)


if __name__ == '__main__':
    tot_time = 0
    args = parse_args()
    if args.cpu == 1:
        print('CPU time: '),
        sys.stdout.flush()
        result_cpu = speed_check('check_cpu.yaml', 19)
        print('%.1fs' % result_cpu)
        tot_time += result_cpu
    if args.gpu == 1:
        print('GPU time: '),
        sys.stdout.flush()
        result_gpu = speed_check('check_gpu.yaml', 225)
        print('%.1fs' % result_gpu)
        tot_time += result_gpu
    if args.dist == 1:
        print('DIST time: '),
        sys.stdout.flush()
        result_dist = speed_check('check_dist.yaml', 60)
        print('%.1fs' % result_dist)
        tot_time += result_dist
        pass
    if tot_time > 0:
        print('total time: %.1fs' % tot_time)
    sys.exit(0)
