# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains various functions and wrappers to make code python 2 and python 3
compatible
"""

import sys
import logging


logger = logging.getLogger(__name__)
PY3 = (sys.version_info[0] >= 3)

# keep range calls consistent between python 2 and 3
# note: if you need a list and not an iterator you can do list(range(x))
range = range
if not PY3:
    logger.info("using xrange as range")
    range = xrange

# keep cPickle, Quque import consistent between python 2 and 3 (where each was
# renamed)
if not PY3:
    import cPickle as the_pickle
    import Queue as the_queue
else:
    import pickle as the_pickle
    import queue as the_queue
pickle = the_pickle
queue = the_queue
