"""
Utility functions for saving various types of objects state.
"""

import cPickle
import logging

logger = logging.getLogger(__name__)


def deserialize(load_path):
    """
    Converts a serialized object into a python data structure.

    Arguments:
        load_path (str): path and name of the serialized on-disk file to load.

    Returns:
        object: Converted in-memory data structure.

    See Also:
        serialize
    """
    logger.info("deserializing object from:  %s" % load_path)
    return cPickle.load(open(load_path))


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
