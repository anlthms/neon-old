# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Utility functions for saving various types of objects state.
"""

import logging
import os
import yaml

from neon.util.compat import PY3, MPI_INSTALLED

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


def extract_child_node_vals(node, keys):
    """
    Helper to iterate through the immediate children of the yaml node object
    passed, looking for the key values specified.

    Arguments:
        node (yaml.nodes.Node): the parent node upon which to being the search
        keys (list): set of strings indicating the child keys we want to
                     extract corresponding values for.

    Returns:
        dict: with one item for each key.  value is value found in search for
              that key, or None if not found.
    """
    res = dict()
    for child in node.value:
        tag, val = [x.value for x in child]
        for key in keys:
            if tag == key:
                res[key] = val
    for key in keys:
        if key not in res:
            res[key] = None
    return res


def obj_multi_constructor(loader, tag_suffix, node,
                          deserialize_param='serialized_path',
                          dist_param='dist_flag'):
    """
    Utility function used to actually import and generate a new class instance
    from its name and parameters, potentially deserializing an already
    serialized representation.

    Arguments:
        loader (yaml.loader.SafeLoader): carries out actual loading
        tag_suffix (str): The latter portion of the tag, representing the full
                          module and class name of the object being
                          instantiated.
        node (yaml.MappingNode): tag/value set specifying the parameters
                                 required for constructing new objects of this
                                 type
        deserialize_param (str): Tag name of the parameter that can be
                                 inspected to deserialize an already existing
                                 instance, instead of constructing a new
                                 object.  Defaults to 'serialized_path'.
        dist_param (str): Tag name of the parameter that can be inspected to
                          indicate the object in question is distributed.
                          Defaults to 'dist_flag'.
    """
    # extract class name and import neccessary module
    parts = tag_suffix.split('.')
    module = '.'.join(parts[:-1])
    cls = __import__(module)
    for comp in parts[1:]:
        cls = getattr(cls, comp)

    # peek at the immediate parameters of this object to see if we should
    # deserialize instead of construct a new object, MPI also requires some
    # special handling
    res = None
    dpath = None
    child_vals = extract_child_node_vals(node, [deserialize_param, dist_param])
    child_vals[dist_param] = (child_vals[dist_param] == 'True')
    if child_vals[deserialize_param] is not None:
        # deserialization attempt should be made
        dpath = child_vals[deserialize_param]
        if ((child_vals[dist_param] and 'datasets' in cls.__module__) or
                ('models' in cls.__module__ and 'Dist' in cls.__name__)):
            # Attempting to deserialize a distributed dataset or model, need to
            # adjust the path from what is specified in the YAML.
            if MPI_INSTALLED:
                from mpi4py import MPI
                dpath = dpath.format(rank=str(MPI.COMM_WORLD.rank),
                                     size=str(MPI.COMM_WORLD.size))
                if os.path.exists(dpath):
                    res = deserialize(dpath)
                    res.__dict__[deserialize_param] = dpath
            else:
                raise AttributeError("dist obj but mpi4py not installed")
        elif os.path.exists(dpath):
            # All other types of deserializable objects
            res = deserialize(dpath)
    if res is None:
        # need to create a new object
        try:
            res = cls(**loader.construct_mapping(node, deep=True))
            if child_vals[deserialize_param] is not None and dpath is not None:
                res.__dict__[deserialize_param] = dpath
        except TypeError as e:
            logger.warning("Unable to construct '%s' instance.  Error: %s",
                           cls.__name__, e.message)
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
    fname = load_path.name
    logger.warn("deserializing object from:  %s", fname)
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
    logger.warn("serializing %s to: %s", str(obj), save_path)
    ensure_dirs_exist(save_path)
    pickle.dump(obj, open(save_path, 'wb'), -1)


class YAMLable(yaml.YAMLObject):
    """
    Base class for any objects we'd like to be able to safely parse from yaml
    configuration strems (or dump suitable representation back out to such a
    stream).
    """
    yaml_loader = yaml.SafeLoader
