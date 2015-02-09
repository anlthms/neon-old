#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Python source code pre-processor to inject/strip specific compiler hints.
"""

import argparse
import sys
import os

__version__ = "0.0.1"

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
indent = '    '
tensor_free_str = (indent + 'def __del__(self):\n' +
                   2 * indent + '"""\n' +
                   2 * indent + 'Called before object destruction.\n' +
                   2 * indent + '"""\n' +
                   2 * indent + 'self.free()\n' +
                   '\n')


def strip_compiler_hints():
    """
    Update source code files in place to remove any traces of existing compiler
    hint code.

    Note that this code makes many assumptions about content!
    """
    # remove free calls from backends
    for be in ["cpu", "gpu"]:
        file_handle = open(os.path.join(src_root, "backends", be + ".py"),
                           'r+')
        content = file_handle.read()
        new_content = content.replace(tensor_free_str, '')
        if new_content != content:
            file_handle.seek(0)
            file_handle.truncate()
            file_handle.write(new_content)
        file_handle.close()

    # remove begin() and end() lines
    for root, dirs, files in os.walk(src_root):
        for fname in files:
            if fname.endswith(".py"):
                file_handle = open(os.path.join(root, fname), 'r+')
                content = file_handle.read()
                new_content = content
                beg_str = "self.backend.begin()"
                end_str = "self.backend.end()"
                if fname in ["cpu.py", "gpu.py"]:
                    beg_str = "self.begin()"
                    end_str = "self.end()"
                for indent_level in [7, 6, 5, 4, 3]:
                    # for or while loop will appear in a function in a file,
                    # hence a minimum of 3 indents for begin/end
                    new_content = new_content.replace(indent_level * indent +
                                                      beg_str + '\n', '')
                    new_content = new_content.replace(indent_level * indent +
                                                      end_str + '\n', '')
                if new_content != content:
                    file_handle.seek(0)
                    file_handle.truncate()
                    file_handle.write(new_content)
                file_handle.close()


def inject_compiler_hints():
    """
    Update source code files in place to insert various compiler hints, useful
    for the system software team.

    Note that this code makes many assumptions about files and content!
    """

    # insert free calls as part of backend Tensor destruction
    for be in ["cpu", "gpu"]:
        file_handle = open(os.path.join(src_root, "backends", be + ".py"),
                           'r+')
        content = file_handle.read()
        # insert in the Tensor class, just before the __str__ function:
        if content.find(tensor_free_str) == -1:
            class_start = content.find("class " + be.upper() + "Tensor")
            fn_start = content.find(indent + "def __str__", class_start)
            content = content[:fn_start] + tensor_free_str + content[fn_start:]
            file_handle.seek(0)
            file_handle.truncate()
            file_handle.write(content)
            file_handle.close()

    # insert begin() and end() lines inside each loop
    for root, dirs, files in os.walk(src_root):
        proceed = False
        for good_dir in ["backends", "layers", "models", "optimizers",
                         "transforms"]:
            if root.endswith(good_dir):
                proceed = True
        if not proceed:
            continue
        for fname in files:
            if fname.endswith(".py"):
                file_handle = open(os.path.join(root, fname), 'r+')
                content = file_handle.read()
                new_content = content
                beg_str = "self.backend.begin()"
                end_str = "self.backend.end()"
                start_idx = 0
                if fname in ["cpu.py", "gpu.py"]:
                    beg_str = "self.begin()"
                    end_str = "self.end()"
                    # cheap hack to skip Tensor classes as they appear first in
                    # these files.
                    start_idx = new_content.find("class " + fname[0:3].upper()
                                                 + "(")
                for indent_level in [6, 5, 4, 3, 2]:
                    for loop_str in ["while ", "for "]:
                        idx = new_content.find('\n' + indent_level * indent +
                                               loop_str, start_idx)
                        ws = (indent_level + 1) * indent
                        ws_len = (indent_level + 1) * len(indent)
                        e_len = (indent_level + 2) * len(indent)
                        while idx != -1:
                            # found the start of a loop, inject begin at start
                            # of next line unless already present
                            idx = new_content.find('\n', idx + 1) + 1
                            if (new_content[idx:idx + ws_len + len(beg_str)]
                                    != (ws + beg_str)):
                                new_content = (new_content[:idx] + ws +
                                               beg_str + '\n' +
                                               new_content[idx:])
                                # now read lines until we find the last one of
                                # this indent_level (skipping empty lines)
                                while (new_content[idx:idx + ws_len] == ws or
                                       new_content[idx:idx + ws_len + 1] ==
                                       ('\n' + ws)):
                                    # if this line contains a continue or break
                                    # statement, we also need to inject another
                                    # end statement
                                    for early_exit in ["continue", "break"]:
                                        if (new_content[idx:idx + e_len +
                                                        len(early_exit)] ==
                                                ws + indent + early_exit):
                                            new_content = (new_content[0:idx] +
                                                           ws + indent +
                                                           end_str + '\n' +
                                                           new_content[idx:])
                                            idx += e_len + len(end_str) + 1
                                    idx = new_content.find('\n', idx) + 1
                                # now inject end at idx and look for the next
                                # loop
                                new_content = (new_content[:idx] + ws +
                                               end_str + '\n' +
                                               new_content[idx:])
                            else:
                                # ensure we advance past the current loop
                                idx += ws_len + len(beg_str)
                            idx = new_content.find('\n' + indent_level * indent
                                                   + loop_str, idx)
                if new_content != content:
                    file_handle.seek(0)
                    file_handle.truncate()
                    file_handle.write(new_content)
                file_handle.close()


def parse_args():
    """
    Sets up and handles command line argument parsing.
    """
    parser = argparse.ArgumentParser(description='Utility to pre-process neon '
                                     'source code files to insert or remove '
                                     'compiler hints that are otherwise '
                                     'too expensive to leave in the code base.'
                                     )
    parser.add_argument('-s', '--strip', action='store_true',
                        help='Attempt to remove any inserted compiler hints')
    parser.add_argument('-v', '--version', action='version',
                        version=__version__)
    return(parser.parse_args())


def main():
    """
    Point of code entry.
    """
    args = parse_args()
    if args.strip:
        strip_compiler_hints()
    else:
        inject_compiler_hints()


if __name__ == '__main__':
    sys.exit(main())
