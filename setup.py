#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

import os
from setuptools import setup, Extension, find_packages, Command
import subprocess


# Define version information
VERSION = '0.6.0'
FULLVERSION = VERSION
write_version = True

try:
    pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                            stdout=subprocess.PIPE)
    (so, serr) = pipe.communicate()
    if pipe.returncode == 0:
        FULLVERSION += "+%s" % so.strip().decode("utf-8")
except:
    pass

if write_version:
    txt = "# " + ("-" * 77) + "\n"
    txt += "# " + "Copyright 2014 Nervana Systems Inc. All rights reserved.\n"
    txt += "# " + ("-" * 77) + "\n"
    txt += "\"\"\"\n%s\n\"\"\"\nVERSION = '%s'\nSHORT_VERSION = '%s'\n"
    fname = os.path.join(os.path.dirname(__file__), 'neon', 'version.py')
    a = open(fname, 'w')
    try:
        a.write(txt % ("Project version information.", FULLVERSION, VERSION))
    finally:
        a.close()

# Define dependencies
dependency_links = []
required_packages = ['numpy>=1.8.1', 'PyYAML>=3.11']


class NeonCommand(Command):
    description = "Passes additional build type options to subsequent commands"
    user_options = [('cpu=', None, 'Add CPU backend related dependencies'),
                    ('gpu=', None, 'Add GPU backend related dependencies'),
                    ('dist=', None, 'Add distributed related dependencies'),
                    ('dev=', None, 'Add development related dependencies')]

    def initialize_options(self):
        self.cpu = "0"
        self.gpu = "0"
        self.dist = "0"
        self.dev = "0"

    def run(self):
        if self.dev == "1":
            self.distribution.install_requires += ['nose>=1.3.0',
                                                   'cython>=0.19.1']
        if self.gpu == "1":
            self.distribution.install_requires += ['cudanet']
            self.distribution.dependency_links += ['http://github.com/'
                                                   'NervanaSystems/'
                                                   'cuda-convnet2.git#'
                                                   'egg=cudanet-0.2']
        if self.dist == "1":
            self.distribution.install_requires += ['mpi4py>=1.3.1']

    def finalize_options(self):
        pass

# use cython to compile extension to .c if installed
use_cython = True
suffix = "pyx"
try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
    suffix = "c"
try:
    import numpy
    np_include_dir = numpy.get_include()
except ImportError:
    np_include_dir = ''
extensions = [Extension('neon.backends.flexpt_dtype',
                        sources=['neon/backends/flexpt_dtype.c'],
                        include_dirs=[np_include_dir]),
              Extension('neon.backends.flexpt_cython',
                        ['neon/backends/flexpt_cython.' + suffix],
                        include_dirs=[np_include_dir])]
if use_cython:
    extensions = cythonize(extensions)

setup(name='neon',
      version=VERSION,
      description='Deep learning library with configurable backends',
      long_description=open('README.md').read(),
      author='Nervana Systems',
      author_email='info@nervanasys.com',
      url='http://www.nervanasys.com',
      license='License :: Other/Proprietary License',
      scripts=['bin/neon'],
      ext_modules=extensions,
      packages=find_packages(),
      install_requires=required_packages,
      cmdclass={'neon': NeonCommand},
      classifiers=['Development Status :: 2 - Pre-Alpha',
                   'Environment :: Console',
                   'Environment :: Console :: Curses',
                   'Environment :: Web Environment',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: Other/Proprietary License',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: ' +
                   'Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: System :: Distributed Computing'])
