# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Sanity check
"""
import argparse
import logging
import os
import sys
from neon.backends import gen_backend
from neon.util.persist import deserialize


def parse_args():
    parser = argparse.ArgumentParser(description='Run sanity check examples')
    parser.add_argument('--cpu', default=0, help='Run CPU sanity check',
                        type=int)
    parser.add_argument('--gpu', default=0, help='Run GPU sanity check',
                        type=int)
    parser.add_argument('--datapar', default=0, type=int,
                        help='Run data parallel sanity check')
    parser.add_argument('--modelpar', default=0, type=int,
                        help='Run model parallel sanity check')
    return parser.parse_args()


def sanity_check(conf_file, result, **be_args):
    experiment = deserialize(os.path.join(dir, conf_file))
    backend = gen_backend(model=experiment.model, **be_args)
    experiment.initialize(backend)
    experiment.run()
    print(float(experiment.model.result))
    assert float(experiment.model.result) == result


if __name__ == '__main__':
    # setup an initial console logger (may be overridden in config)
    logging.basicConfig(level=40)  # ERROR or higher
    res = 0
    args = parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    check_file = os.path.join(script_dir, '..', '..', 'examples',
                              'convnet', 'synthetic-sanity_check.yaml')
    expected_result = 0.4921875
    # TODO: modelpar currently broken on synthetic-sanity_check.yaml
    # (dimensions not aligned), so skipping for the moment.
    # for be in ["cpu", "gpu", "datapar", "modelpar"]:
    for be in ["cpu", "gpu", "datapar"]:
        be_args = {'rng_seed': 0}
        if args.__dict__[be] == 1:
            if be != "cpu":
                be_args[be] = 1
            print('{} check '.format(be)),
            sanity_check(check_file, expected_result, **be_args)
            print('OK')
    sys.exit(res)
