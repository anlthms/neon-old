# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Speed check
"""
import argparse
import logging
import os
import sys
import time

from neon.backends import gen_backend
from neon.util.persist import deserialize


def parse_args():
    parser = argparse.ArgumentParser(description='Run speed check examples')
    parser.add_argument('--cpu', default=0, help='Run CPU speed check',
                        type=int)
    parser.add_argument('--gpu', default=0, help='Run GPU speed check',
                        type=int)
    parser.add_argument('--datapar', default=0, type=int,
                        help='Run data parallel speed check')
    parser.add_argument('--modelpar', default=0, type=int,
                        help='Run model parallel speed check')
    return parser.parse_args()


def speed_check(conf_file, num_epochs, **be_args):
    experiment = deserialize(os.path.join(dir, conf_file))
    experiment.model.num_epochs = num_epochs
    backend = gen_backend(model=experiment.model, **be_args)
    experiment.initialize(backend)
    start = time.time()
    experiment.run()
    return (time.time() - start)


if __name__ == '__main__':
    tot_time = 0
    args = parse_args()
    # setup an initial console logger (may be overridden in config)
    logging.basicConfig(level=40)  # ERROR or higher
    script_dir = os.path.dirname(os.path.realpath(__file__))
    check_file = os.path.join(script_dir, '..', '..', 'examples',
                              'convnet', 'synthetic-sanity_check.yaml')
    # NOTE: number of epochs calibrated to take ~30 seconds total on 2014 MBP
    # TODO: modelpar currently broken on synthetic-sanity_check.yaml
    # (dimensions not aligned), so skipping for the moment.
    for be, num_epochs in [("cpu", 120), ("gpu", 225), ("datapar", 120)]:
        be_args = {'rng_seed': 0}
        if args.__dict__[be] == 1:
            if be != "cpu":
                be_args[be] = 1
            if be == "gpu":
                be_args[be] = "cudanet"
            if be == "datapar":
                # temporary hack because we are not running via mpirun.
                os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = '0'
                os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'] = '1'
            print('{} time: '.format(be)),
            sys.stdout.flush()
            res_time = speed_check(check_file, num_epochs, **be_args)
            print('{:.1f}s'.format(res_time))
            tot_time += res_time
    if tot_time > 0:
        print('total time: {:.1f}s'.format(tot_time))
    sys.exit(0)
