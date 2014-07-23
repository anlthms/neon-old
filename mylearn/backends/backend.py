"""
Houses low-level code for performing underlying data manipulation operations.
"""

import logging
import yaml

logger = logging.getLogger(__name__)


class Backend(yaml.YAMLObject):
    """
    Abstract backend defines operations that must be supported.

    Inherits from yaml.YAMLObject, typically you would utilize a concrete
    child of this class.

    Attributes:
        yaml_loader (yaml.SafeLoader): parser used to load backend.
    """
    yaml_loader = yaml.SafeLoader

    class Tensor(object):
        """
        Represents an arbitrary n-dimensional array data structure.

        Arguments:
            object (numpy.ndarray): An in-memory n-dimensional array containing
                                    the data values.
            dtype (numpy.dtype, optional): The underlying type of each element

        Raises:
            NotImplmentedError: Can't be instantiated directly.
        """
        def __init__(self, object, dtype=None):
            raise NotImplementedError()
