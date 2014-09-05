"""
Contains various functions and wrappers to make code python 2 and python 3
compatible, as well as indicate the presence of a CUDA compatible GPU (or at
least the CUDA SDK).
"""

import os
import sys

PY3 = (sys.version_info[0] >= 3)

CUDA_GPU = False
if sys.platform.startswith("linux"):
    CUDA_GPU = (os.system("nvidia-smi > /dev/null 2>&1") == 0)
elif sys.platform.startswith("darwin"):
    CUDA_GPU = (os.system("kextstat | grep -i cuda > /dev/null 2>&1") == 0)
