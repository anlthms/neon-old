"""
spearmint script: config.pb mentions name: "spear_wrapper" so spearmint calls
into this file's main() function with the current set of parameters. It does:
- read the iris hyper-yaml file
- parse the parameters suggested by sparemint
- generate a temp yaml file
- run neon
- get the outputs
"""

import os, sys, time
from ipdb import set_trace as trace

def main(job_id, params):
    print 'main: job #:', str(job_id)
    print "main: we are in directory: ", os.getcwd()
    print "main: params are: ", params

    return call_convnet(params)

def call_convnet(params): #
    """
    runs the system call to neon and reads the result to give back to sm
    """

    timestring = str(int(time.time()))
    basepath = '/Users/urs/code/neon/'
    hyperyaml = 'hyper_iris_cpu_mlp-4-2-3.yaml'

    # Generate the yaml ifle
    print "call_convnet: parsing yaml"
    input_file = basepath + '/examples/' + hyperyaml
    output_file = 'yamels/temp'+timestring+'.yaml'
    try:
        os.mkdir('yamels')
    except OSError:
        "Directory exists"
    write_params(input_file, output_file, params)

    # Run a model
    print "call_convnet: running neon"
    callstring = basepath + "bin/neon " + output_file
    os.system(callstring)

    # Read the model output
    print "call_convnet: writing outputs"
    with open('neon_result_validation.txt', 'r') as f:
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
                    print "write_params: trying to set a trace"
                    line = parse_line(line, params)
                    print "line:", line
                fout.write(line)

def parse_line(line, params):
    """
    Replace the line defining the parameter range by just a name value pair.
    """
    dic = [k.strip("{},") for k in line.split()]
    out = params[dic[2]][0]

    return dic[0] + " " + str(out) + ",\n"
