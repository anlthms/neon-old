"""
Utility functions for saving various types of objects state.
"""

import cPickle
import logging

import yaml

logger = logging.getLogger(__name__)

# ensure yaml constructors and so forth get registered prior to first load
# attempt.
yaml_initialized = False


def obj_multi_constructor(loader, tag_suffix, node):
    """
    Utility function used to actually import and generate a new class instance
    from its name and parameters.

    Arguments:
        loader (yaml.loader.SafeLoader): carries out actual loading
        tag_suffix (str): The latter portion of the tag, representing the full
                          module and class name of the object being
                          instantiated.
        node (yaml.MappingNode): tag/value set specifying the parameters
                                 required for constructing new objects of this
                                 type
    """

    # extract class name and import neccessary module
    parts = tag_suffix.split('.')
    module = '.'.join(parts[:-1])
    cls = __import__(module)
    for comp in parts[1:]:
        cls = getattr(cls, comp)

    # get they key/value pairs from node and instantiate the object
    try:
        res = cls(**loader.construct_mapping(node, deep=True))
    except TypeError as e:
        logger.warning("Unable to construct '%s' instance.  Error: %s" %
                       (cls.__name__, e.message))
        res = None
    return res


def initialize_yaml():
    global yaml_initialized
    yaml.add_multi_constructor('!obj:', obj_multi_constructor,
                               yaml.loader.SafeLoader)
    yaml_initialized = True


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
    global yaml_initialized
    if not isinstance(load_path, file):
        load_path = file(load_path)
    logger.info("deserializing object from:  %s" % load_path.name)
    if (load_path.name.lower().endswith('.yaml') or 
        load_path.name.lower().endswith('.yml')):
        if not yaml_initialized:
            initialize_yaml()
        return yaml.safe_load(load_path)
    else:
        return cPickle.load(load_path)


def serialize(obj, save_path):
    """
    Dumps a python data structure to a saved on-disk representation.  We
    currently support writing to the following file formats (expected filename
    extension in brackets):
    
        * python pickle (.pkl)

    Arguments:
        obj (object): the python object to be saved.
        save_path (str): Where to write the serialized object (full path and
                         file name)

    See Also:
        deserialize
    """
    logger.info("serializing %s to: %s" % (str(obj), save_path))
    cPickle.dump(obj, open(save_path, 'wb'), -1)


class YAMLable(yaml.YAMLObject):
    """
    Base class for any objects we'd like to be able to safely parse from yaml
    configuration strems (or dump suitable representation back out to such a
    stream).
    """
    yaml_loader = yaml.SafeLoader
