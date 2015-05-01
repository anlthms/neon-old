.. ---------------------------------------------------------------------------
.. Copyright 2014 Nervana Systems Inc.  All rights reserved.
.. ---------------------------------------------------------------------------

Frequently Asked Questions
--------------------------

Installation
============

* Does neon run on Microsoft Windows?

  * At this time we are only supporting Linux and Mac OSX based installations.

* During install on a Mac I get "yaml.h" file not found error when the PyYAML
  package is being built.  Is that bad?

  * It can safely be ignored in this situation.  The problem is that you don't
    have libyaml installed so PyYAML will resort to its own (slightly slower)
    implementation. Without it, neon will still be able to successfully parse
    and read your Experiment's YAML files.


Running
=======

* The console output is too verbose, how do I reduce the amount of logging?

  * In the YAML file for your experiment, you can reduce the amount of logging
    by increasing the numeric value of ``logging``'s ``level`` parameter.  A
    value of 40 implies that only messages of type ERROR and CRITICAL will be
    displayed for instance.

* The console output is too sparse, how do I increase the amount of logging?

  * In the YAML file for your experiment, you can increase the amount of logging
    by decreasing the numeric value of ``logging``'s ``level`` parameter.  A
    value of 10 implies that messages of type DEBUG, INFO, WARNING, ERROR, and
    CRITICAL will all be displayed for instance.

Contributing
============

* I think I found a bug, what do I do?

  * Please search our
    `Github issues <https://github.com/NervanaSystems/neon/issues>`_ list and 
    if it hasn't already been addressed, file a new issue so we can take a
    look.
