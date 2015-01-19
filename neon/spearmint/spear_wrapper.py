"""
spearmint script: config.pb refers to "spear_wrapper" so spearmint calls
into this file's main() function with the current set of parameters. It then:
- reads the hyper-yaml file
- parses the parameters suggested by sparemint
- generates a temp yaml file
- runs neon
- gets the outputs
"""

import os
import time
import logging

logger = logging.getLogger(__name__)


def main(job_id, params):
    logger.info('spear_wrapper job #:%s' % str(job_id))
    logger.info("spear_wrapper in directory: %s" % os.getcwd())
    logger.info("spear_wrapper params are:%s" % params)

    return call_convnet(params)


def call_convnet(params):
    """
    runs the system call to neon and reads the result to give back to sm
    """
    timestring = str(int(time.time()))

    # Generate the yaml file
    hyper_file = 'hyperyaml.yaml'
    yaml_file = 'yamels/temp'+timestring+'.yaml'
    try:
        os.mkdir('yamels')
    except OSError:
        "Directory exists"
    result_fname = write_params(hyper_file, yaml_file, params)

    # System call to run bin/neon model
    callstring = "../../bin/neon " + yaml_file
    os.system(callstring)

    # Read the model output error from txt file
    with open(result_fname, 'r') as f:
            result = map(float, f)
    return result[0]


def write_params(input_file, output_file, params):
    """
    go thorugh the hyperyaml line by line to create tempyaml
    """
    with open(input_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                if 'hyperopt' in line:
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
