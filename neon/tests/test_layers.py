"""
Sanity test layers.
"""
import os
import neon
import logging
from neon.util.persist import deserialize


def run_sanity(conf_file, result):
    dir = os.path.dirname(os.path.realpath(__file__))
    experiment = deserialize(os.path.join(dir, conf_file))
    if hasattr(experiment, 'logging'):
        logging.basicConfig(**experiment.logging)

    experiment.run()
    assert experiment.model.result == result


def test_layers():
    run_sanity('sanity_cpu.yaml', 0.3203125)
    # XXX: temporarily commented out due to assertion failure in initCublas()
    #run_sanity('sanity_gpu.yaml', 0.390625)

