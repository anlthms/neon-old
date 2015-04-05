# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
hyperopt script: spearmint calls into this file's main() function with the
current set of hyperparameters (selected by spearmint). It then:
- reads the hyper-yaml file
- parses the parameters suggested by spearmint
- generates a temp yaml file
- runs neon
- gets the outputs
"""

import os
import time
import logging
from neon.backends import gen_backend
from neon.util.persist import deserialize


def main(job_id, params):
    print('spear_wrapper job #:%s' % str(job_id))
    print("spear_wrapper in directory: %s" % os.getcwd())
    print("spear_wrapper params are:%s" % params)

    return call_neon(params)


def call_neon(params):
    """
    runs the system call to neon and reads the result to give back to sm
    """
    timestring = str(int(time.time()))
    experiment_dir = os.path.realpath(os.environ['HYPEROPT_PATH'])
    # Generate the yaml file
    hyper_file = os.path.join(experiment_dir, 'hyperyaml.yaml')
    yaml_file = os.path.join(experiment_dir, 'yamels',
                             'temp' + timestring + '.yaml')
    try:
        os.mkdir('yamels')
    except OSError:
        "Directory exists"
    result_fname = write_params(hyper_file, yaml_file, params)

    # run bin/neon model
    logging.basicConfig(level=20)
    experiment = deserialize(yaml_file)
    backend = gen_backend(model=experiment.model)  # , gpu='nervanagpu'
    experiment.initialize(backend)
    return_err = experiment.run()

    return float(return_err)

def write_params(input_file, output_file, params):
    """
    go thorugh the hyperyaml line by line to create tempyaml
    """
    with open(input_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                if '!hyperopt' in line:
                    line = parse_line(line, params)
                if 'filename' in line:
                    retval = line.split()[1].strip(",")
                fout.write(line)
    return retval


def parse_line(line, params):
    """
    Replace the line defining the parameter range by just a name value pair.
    """
    dic = [k.strip("{},") for k in line.split()]
    out = params[dic[2]][0]
    return dic[0] + " " + str(out) + ",\n"
