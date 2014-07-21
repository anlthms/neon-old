"""
Contains miscellaneous classes and functionality.
"""

import logging

logger = logging.getLogger(__name__)


class Factory(object):
    """
    Abstract factory used to instantiate an object of a particular class.
    """

    @staticmethod
    def _get_class(cls_str):
        """
        Helper to map an input string to a class instance

        :param cls_str: name of the class
        :type cls_str: str
        :returns: class reference
        :raises: ImportError
        """
        parts = cls_str.split('.')
        module = '.'.join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m

    @staticmethod
    def create(**kwargs):
        """
        Object construction method.

        :param type: type of object to create
        :type type: str
        :returns: instance of class specified by the type arg (if valid)
        """
        if 'type' in kwargs:
            cls = Factory._get_class(kwargs['type'])
            return cls(**kwargs)
