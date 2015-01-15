"""
spearmint script: config.pb mentions name: "spear_wrapper" so spearmint calls
into this file's main() function with the current set of parameters. It does:
- read the iris hyper-yaml file
- parse the parameters suggested by sparemint
- generate a temp yaml file
- run neon
- get the outputs
"""

import os
#os.chdir('/home/local/code/cuda-convnet2/')
import sys
import time
#sys.path.append('/home/local/code/cuda-convnet2/')
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

    params['mom'] = 0.9
    timestring = str(int(time.time()))

    # Part 1: generate the yaml ifle
    print "call_convnet: parsing yaml"
    input_file = '/Users/urs/code/neon/examples/hyper_iris_cpu_mlp-4-2-3.yaml'
    output_file = 'yamels/temp'+timestring+'.yaml'
    log_file = 'log_it.txt'
    write_params(input_file, output_file, log_file, params)

    # Part 2: Run a model
    print "call_convnet: running neon"
    basepath = '/Users/urs/code/neon/'
    callstring = basepath + "bin/neon " + output_file
    retval = os.system(callstring)

    # Part 3: Read the model output
    print "call_convnet: writing outputs"
    with open('neon_result_validation.txt', 'r') as f:
            result = map(float, f)
    return result[0]

def write_params(input_file, output_file, log_file, params):
    """
    go thorugh the hyperyaml line by line, fill in proposed values from sm
    and write line by line to tempyaml
    [TODO] right now this is a hardcoded set of parameters. Ultimately we
           want to adapt this to what the yaml is "requesting" from sm.
    """
    #mom =  params['mom']  # 0.9 hardcoded above
    #nin1, nin2 = params['numberneurons'][0], params['numberneurons'][1]
    #step = params['stepsize']

    # read all lines from source
    with open(input_file, 'r') as f:
        # write all lines to target
        with open(output_file, 'w') as g:
            # append to log file
            with open(log_file, 'a') as h:
                for line in f:
                    if 'hyperopt' in line:
                        print "write_params: trying to set a trace"
                        line = parse_line(line, params)
                        print "line:", line
                        h.write(": " + line)
                    # write all lines, but parse ones with hyperopt.
                    g.write(line)

def parse_line(line, params):
    """
    This function should strip off the "hyperopt" and instead place in a
    value
    """
    dic = [k.strip("{},") for k in line.split()]
    print "parseline: split into", dic
    # check sanity
    if (dic[2] != 'numberneurons') and (dic[2] != 'stepsize')and (dic[2] != 'epochs'):
        # want number_neurons and learning_rates, nothing else is supported.
        raise NameError('UrsIsNotCoolWithThisError')

    # ho = dict()
    # ho['chooser'] = dic[1].split(':')[1]
    # ho['type'] = dic[2]
    # ho['end'] = float(dic[3])
    # ho['start'] = float(dic[4])
    # print "parseline: ho", ho

    out = params[dic[2]][0] # need to figure out which one to pick here!

    return dic[0] + " " + str(out) + ",\n"
