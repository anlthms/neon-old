# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Defines how to collect components together to run an experiment.
"""

from neon.util.persist import YAMLable


class Experiment(YAMLable):
    """
    Abstract base class definining the required interface for each concrete
    experiment.

    All items that are required to define an experiment (models, datasets, and
    so forth) should be passed in as keyword arguments to the constructor.
    This inherits configuration file handling via `yaml.YAMLObject
    <http://pyyaml.org/wiki/PyYAMLDocumentation#YAMLObject>`_
    """

    def run(self):
        """
        The method that gets called to actually carry out all the steps of an
        experiment.

        Raises:
            NotImplementedError: Create a concrete child class and implement
                                 this method there.
        """
        raise NotImplementedError()
