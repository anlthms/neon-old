# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Script for generating the PB file for Spearmint:
- Go through the hyper-yaml
- extract lines that specify a !hyperopt range
- format and dump an entry into the protobuf
"""

import os


def write_pb(input_file, pb_file):
    # go thorugh the hyperyaml line by line, read out values and write to pb
    scipt_name = 'spear_wrapper'  # script spearmint should call
    supported_expt_bool = False  # hyperyaml specifies supported experiment
    with open(input_file, 'r') as fin:
        with open(pb_file, 'w') as fout:
            fout.write('language: PYTHON \nname: "' + scipt_name + '"\n\n')
            for inline in fin:
                if 'hyperopt' in inline:
                    ho_dict = parse_line(inline)
                    outline = write_block(ho_dict)
                    fout.write(outline)
                if 'WriteErrorToFile' in inline:
                    supported_expt_bool = True
    return supported_expt_bool


def parse_line(line):
    # generate a dictionary ho_dict with fields: [name, type, start, end]
    dic = [k.strip("{},") for k in line.split()]
    ho_dict = dict()
    ho_dict['name'] = dic[2]
    ho_dict['type'] = dic[3]
    if (ho_dict['type'] == 'FLOAT'):
        ho_dict['end'] = float(dic[4])
        ho_dict['start'] = float(dic[5])
    elif (ho_dict['type'] == 'INT'):
        ho_dict['end'] = int(dic[4])
        ho_dict['start'] = int(dic[5])
    elif (ho_dict['type'] == 'STRING'):
        ho_dict['end'] = dic[4]
        ho_dict['start'] = dic[5]
    else:
        print "got ho_dict['type']", ho_dict['type']
        raise AttributeError("Supported types are FLOAT, INT, ENUM")
        # todo: Spearmint supports ENUM but we are not handling it yet.
    return ho_dict


def write_block(ho_dict):
    # generate a block for the protobuf file from the hyperopt parameters
    outline = """variable {
    name: \""""+ho_dict['name']+"""\"
    type: """+ho_dict['type']+"""
    size: 1
    min:  """+str(ho_dict['start'])+"""
    max:  """+str(ho_dict['end'])+"""
    }\n\n"""
    return outline


def main():
    # point of code entry
    hyperroot = os.path.dirname(os.path.realpath(__file__))
    in_file = os.path.join(hyperroot, 'hyperyaml.yaml')
    pb_file = os.path.join(hyperroot, 'spear_config.pb')

    success = write_pb(in_file, pb_file)
    if success:
        print "Done writing hyper ranges from ", in_file, "to", pb_file
    else:
        raise AttributeError("Wrong experiment type, does not return result")
