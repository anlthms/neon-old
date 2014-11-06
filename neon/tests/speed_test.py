"""
Speed test
"""
import os
import sys
import time
import neon
import logging
from neon.util.persist import deserialize


def speed_test(conf_file, num_epochs):
    dir = os.path.dirname(os.path.realpath(__file__))
    experiment = deserialize(os.path.join(dir, conf_file))
    experiment.model.num_epochs = num_epochs 
    if hasattr(experiment, 'logging'):
        logging.basicConfig(**experiment.logging)

    start = time.time()
    experiment.run()
    return (time.time() - start)


if __name__ == '__main__':
    result_cpu = speed_test('sanity_cpu.yaml', 19)
    result_gpu = speed_test('sanity_gpu.yaml', 225)
    print 'cpu time: %.1fs, gpu time: %.1fs, total: %.1fs' % (
        result_cpu, result_gpu, result_cpu + result_gpu)
