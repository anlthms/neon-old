"""
Defines how to collect components together to run an experiment.
"""

import yaml


class Experiment(yaml.YAMLObject):
    """
    Abstract base class definining the required interface for each concrete
    experiment.

    All items that are required to define an experiment (models, datasets, and
    so forth) should be passed in as keyword arguments to the constructor.
    This inherits configuration file handling via `yaml.YAMLObject
    <http://pyyaml.org/wiki/PyYAMLDocumentation#YAMLObject>`_

    Attributes:
        yaml_loader (yaml.SafeLoader): parser used to load experiment.
    """
    yaml_loader = yaml.SafeLoader

    def run(self):
        """
        The method that gets called to actually carry out all the steps of an
        experiment.

        Raises:
            NotImplementedError: Create a concrete child class and implement
                                 this method there.
        """
        raise NotImplementedError()
