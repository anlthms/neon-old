#!/usr/bin/env python

from distutils.core import setup
import os


# Define version information
VERSION = '0.0.0'
FULLVERSION = VERSION
write_version = True

if write_version:
    txt = "version = '%s'\nshort_version = '%s'\n"
    fname = os.path.join(os.path.dirname(__file__), 'mylearn', 'version.py')
    a = open(fname, 'w')
    try:
        a.write(txt % (FULLVERSION, VERSION))
    finally:
        a.close()

setup(name='mylearn',
      version=VERSION,
      description='Deep learning library with configurable backends',
      author='Nervana Systems',
      author_email='software@nervanasys.com',
      url='http://www.nervanasys.com',
      packages=['mylearn',
                'mylearn.backends',
                'mylearn.datasets', 
                'mylearn.experiments', 
                'mylearn.models',
                'mylearn.util',
                'mylearn.tests',
                'mylearn.backends.tests',
                'mylearn.datasets.tests', 
                'mylearn.experiments.tests', 
                'mylearn.models.tests',
                'mylearn.util.tests',
               ],
      scripts=['bin/mylearn'],
     )
