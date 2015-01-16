
"""
generating the PB file for Spearmint: Go through the hyper-yaml, extract lines
that specify a !hyperopt range, dump an entry into the protobuf
"""

import os, sys, time
from ipdb import set_trace as trace

def write_pb(input_file, pb_file):
    """
    go thorugh the hyperyaml line by line, read out values and write to pb
    """
    scipt_name = 'spear_wrapper'  # script spearmint should call
    we_are_good = False
    # read all lines from source
    with open(input_file, 'r') as fin:
        # append to pb file
        with open(pb_file, 'w') as fout:
            fout.write('language: PYTHON \nname: "' + scipt_name + '"\n\n')
            for inline in fin:
                if 'hyperopt' in inline:
                    ho = parse_line(inline)
                    outline = write_line(ho)
                    fout.write(outline)
                if 'WriteErrorToFile' in inline:
                    we_are_good = True
    return we_are_good

def parse_line(line):
    """
    generate a dictionary ho with fields:
    name
    type
    start
    end
    """
    dic = [k.strip("{},") for k in line.split()]
    ho = dict()
    ho['name'] = dic[2]
    ho['type'] = dic[3]
    if (ho['type'] == 'FLOAT'):
        ho['end'] = float(dic[4])
        ho['start'] = float(dic[5])
    elif (ho['type'] == 'INT'):
        ho['end'] = int(dic[4])
        ho['start'] = int(dic[5])
    elif (ho['type'] == 'STRING'):
        ho['end'] = dic[4]
        ho['start'] = dic[5]
    else:
        print "got ho['type']", ho['type']
        raise AttributeError("Supported types are FLOAT, INT, STRING")

    return ho

def write_line(ho):
    """
    stuff
    """
    outline = """variable {
    name: \""""+ho['name']+"""\"
    type: """+ho['type']+"""
    size: 1
    min:  """+str(ho['start'])+"""
    max:  """+str(ho['end'])+"""
    }\n\n"""
    return outline

if __name__=='__main__':
    """
    specify hyperyaml to read from and protobuf to write to
    """

    # point of entry
    input_file = 'neon/spearmint/hyperyaml.yaml'
    pb_file = 'neon/spearmint/spear_config.pb'

    success = write_pb(input_file, pb_file)
    if success:
        print "Done writing hyper ranges from ", input_file, "to", pb_file
    else:
        raise AttributeError("Wrong experiment type, does not return result")