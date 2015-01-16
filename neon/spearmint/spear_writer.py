
"""
generating the PB file for Spearmint: Go through the hyper-yaml, extract lines
that specify a !hyperopt range, dump an entry into the protobuf
"""


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
                    ho_dict = parse_line(inline)
                    outline = write_line(ho_dict)
                    fout.write(outline)
                if 'WriteErrorToFile' in inline:
                    we_are_good = True
    return we_are_good


def parse_line(line):
    """
    generate a dictionary ho_dict with fields:
    name
    type
    start
    end
    """
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
        raise AttributeError("Supported types are FLOAT, INT, STRING")

    return ho_dict


def write_line(ho_dict):
    """
    stuff
    """
    outline = """variable {
    name: \""""+ho_dict['name']+"""\"
    type: """+ho_dict['type']+"""
    size: 1
    min:  """+str(ho_dict['start'])+"""
    max:  """+str(ho_dict['end'])+"""
    }\n\n"""
    return outline

if __name__ == '__main__':
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
