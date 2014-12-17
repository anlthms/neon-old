# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Overrides persist for hyperoptimization yaml files.
"""

import logging
import os
import yaml
from ipdb import set_trace as trace
from neon.experiments.hyperopt import get_parameters

from neon.util.compat import PY3

if PY3:
    import pickle
else:
    import cPickle as pickle

logger = logging.getLogger(__name__)

# ensure yaml constructors and so forth get registered prior to first load
# attempt.
yaml_initialized = False


def ensure_dirs_exist(path):
    """
    Simple helper that ensures that any directories specified in the path are
    created prior to use.

    Arguments:
        path (str): the path (may be to a file or directory).  Any intermediate
                    directories will be created.

    Returns:
        str: The unmodified path value.
    """
    outdir = os.path.dirname(path)
    if outdir is not '' and not os.path.isdir(outdir):
        os.makedirs(outdir)
    return path

def range_multi_constructor(loader, tag_suffix, node):
    """
    experimental function to include parameter ranges in the yaml files.
    returns a dict with type, start and end of range
    """

    # get they key/value pairs from node and instantiate the object
    try:
        res = loader.construct_mapping(node)
    except TypeError as e:
        logger.warning("Unable to construct '%s' instance.  Error: %s" %
                       (cls.__name__, e.message))
        res = None
    res['type'] = tag_suffix.encode('ascii','ignore')

    # cast range to number
    out = get_parameters(res)

    return out

def obj_dummy_constructor(loader, tag_suffix, node):
    """
    do nothing -- obj should be ignored and passed through.
    """
    try:
        prefix = "@@!obj:" + tag_suffix.encode('ascii','ignore') + "@@"
        res = loader.construct_mapping(node)
    except TypeError as e:
        logger.warning("Unable to construct '%s' instance.  Error: %s" %
                       (cls.__name__, e.message))
        res = None
    print "tag_suffix", prefix
    return {prefix: res}

def initialize_yaml():
    global yaml_initialized
    yaml.add_multi_constructor('!range:', range_multi_constructor,
                               yaml.loader.SafeLoader)
    yaml.add_multi_constructor('!obj:', obj_dummy_constructor,
                               yaml.loader.SafeLoader)
    yaml_initialized = True


def deserialize_and_cast(load_path):
    """
    Converts a serialized object into a python data structure.  We currently
    support reading from the following file formats (expected filename
    extension in brackets):

        * python pickle (.pkl)
        * YAML (.yaml)

    Arguments:
        load_path (str, File): path and name of the serialized on-disk file to
                               load (or an already loaded file object).
                               The type to write is inferred based on filename
                               extension.  If no extension given, pickle format
                               is attempted.

    Returns:
        object: Converted in-memory python data structure.

    See Also:
        serialize
    """
    global yaml_initialized
    if not isinstance(load_path, file):
        load_path = file(load_path)
    fname = load_path.name
    logger.info("deserializing object from:  %s", fname)
    if (fname.lower().endswith('.yaml') or fname.lower().endswith('.yml')):
        if not yaml_initialized:
            initialize_yaml()
        return yaml.safe_load(load_path)
    else:
        try:
            return pickle.load(load_path)
        except AttributeError:
            msg = ("Problems deserializing: %s.  Its possible the interface "
                   "for this object has changed since being serialized.  You "
                   "may need to remove and recreate it." % load_path)
            logger.error(msg)
            raise AttributeError(msg)


def serialize(obj, save_path):
    """
    Dump the generated yaml file
    """
    logger.info("serializing %s to: %s", str(obj), save_path)
    ensure_dirs_exist(save_path)
    with open(save_path, 'w') as outfile:
        outfile.write(yaml.dump(obj, default_flow_style=True))


class YAMLable(yaml.YAMLObject):
    """
    Base class for any objects we'd like to be able to safely parse from yaml
    configuration strems (or dump suitable representation back out to such a
    stream).
    """
    yaml_loader = yaml.SafeLoader
