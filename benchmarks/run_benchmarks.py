#!/usr/bin/env python
"""
Run all benchmark scripts from this directory sequentially.
"""

import os


def get_scripts():
    scripts = []
    this_dir = os.path.dirname(os.path.realpath(__file__))
    for fname in os.listdir(this_dir):
        if fname.startswith("bench"):
            scripts.append(os.path.join(this_dir, fname))
    return scripts

if __name__ == '__main__':
    # TODO: change this code so that we're not spawning separate processes
    # (utilize some common benchmark class or function definition)
    for script in get_scripts():
        print "Executing %s..." % script
        os.system(script)
