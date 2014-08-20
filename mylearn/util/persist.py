"""
Utility functions for saving various types of objects state.
"""

import cPickle
import logging

import yaml

logger = logging.getLogger(__name__)


def deserialize(load_path):
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
    if not isinstance(load_path, file):
        load_path = file(load_path)
    logger.info("deserializing object from:  %s" % load_path.name)
    if (load_path.name.lower().endswith('.yaml') or 
        load_path.name.lower().endswith('.yml')):
        return yaml.load(load_path)
    else:
        return cPickle.load(load_path)


def serialize(obj, save_path):
    """
    Dumps a python data structure to a saved on-disk representation.

    Arguments:
        obj (object): the python object to be saved.
        save_path (str): Where to write the serialized object (full path and
                         file name)

    See Also:
        deserialize
    """
    logger.info("serializing %s to: %s" % (str(obj), save_path))
    cPickle.dump(obj, open(save_path, 'wb'), -1)
