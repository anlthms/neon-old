#!/usr/bin/env python

import numpy as np
import os
from setuptools import setup, Extension
import subprocess


# Define version information
VERSION = '0.4.0'
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
    fname = os.path.join(os.path.dirname(__file__), 'neon', 'version.py')
    a = open(fname, 'w')
    try:
        a.write(txt % ("Project version information.", FULLVERSION, VERSION))
    finally:
        a.close()

use_cython = True
suffix = "pyx"
try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
    suffix = "c"
extensions = [Extension('neon.backends.fixpt_dtype',
                        sources=['neon/backends/fixpt_dtype.c'],
                        include_dirs=[np.get_include()]),
              Extension('neon.backends.fixpt_cython',
                        ['neon/backends/fixpt_cython.' + suffix],
                        include_dirs=[np.get_include()])]
if use_cython:
    extensions = cythonize(extensions)

setup(name='neon',
      version=VERSION,
      description='Deep learning library with configurable backends',
      long_description=open('README.md').read(),
      author='Nervana Systems',
      author_email='info@nervanasys.com',
      url='http://www.nervanasys.com',
      packages=['neon',
                'neon.backends',
                'neon.datasets',
                'neon.experiments',
                'neon.models',
                'neon.transforms',
                'neon.util',
                'neon.util.distarray',
                'neon.tests',
                'neon.backends.tests',
                'neon.datasets.tests',
                'neon.experiments.tests',
                'neon.models.tests',
                'neon.transforms.tests',
                'neon.util.tests', ],
      scripts=['bin/neon'],
      ext_modules=extensions)
