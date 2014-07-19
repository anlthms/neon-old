"""
Houses low-level code for performing underlying data manipulation operations.
"""

import logging
import yaml

logger = logging.getLogger(__name__)


class Backend(yaml.YAMLObject):
    """
    Abstract backend defines operations that must be supported.
    """
    yaml_loader = yaml.SafeLoader

    class Tensor(object):
        """
        Represents an n-dimensional array.
        """
        def __init__(self, object, dtype=None):
            """
            object should be an in-memory n-dimensional array.
            """
            raise NotImplementedError()
