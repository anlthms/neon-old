#!/usr/bin/env python

import numpy as np
import os
from setuptools import setup, Extension
import subprocess


# Define version information
VERSION = '0.2.0'
FULLVERSION = VERSION
write_version = True

git_rev = None
try:
    pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                            stdout=subprocess.PIPE)
    (so, serr) = pipe.communicate()
    if pipe.returncode == 0:
        FULLVERSION += "+%s" % so.strip()
except:
    pass

if write_version:
    txt = "\"\"\"\n%s\n\"\"\"\nVERSION = '%s'\nSHORT_VERSION = '%s'\n"
    fname = os.path.join(os.path.dirname(__file__), 'mylearn', 'version.py')
    a = open(fname, 'w')
    try:
        a.write(txt % ("Project version information.", FULLVERSION, VERSION))
    finally:
        a.close()

setup(name='mylearn',
      version=VERSION,
      description='Deep learning library with configurable backends',
      long_description=open('README.md').read(),
      author='Nervana Systems',
      author_email='software@nervanasys.com',
      url='http://www.nervanasys.com',
      packages=['mylearn',
                'mylearn.backends',
                'mylearn.datasets',
                'mylearn.experiments',
                'mylearn.models',
                'mylearn.transforms',
                'mylearn.util',
                'mylearn.tests',
                'mylearn.backends.tests',
                'mylearn.datasets.tests',
                'mylearn.experiments.tests',
                'mylearn.models.tests',
                'mylearn.transforms.tests',
                'mylearn.util.tests', ],
      scripts=['bin/mylearn'],
      ext_modules=[
          Extension('mylearn.backends.fixpt_dtype',
                    sources=['mylearn/backends/fixpt_dtype.c'],
                    include_dirs=[np.get_include()])])
