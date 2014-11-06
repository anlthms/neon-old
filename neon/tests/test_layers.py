"""
Sanity test layers.
"""
import os
import neon
import logging
from neon.util.persist import deserialize


def test_layers():
    dir = os.path.dirname(os.path.realpath(__file__))
    experiment = deserialize(os.path.join(dir, 'sanity_layers.yaml'))
    if hasattr(experiment, 'logging'):
        logging.basicConfig(**experiment.logging)

    experiment.run()
    assert experiment.model.result == 0.3203125
