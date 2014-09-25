"""
mylearn - A highly configurable machine learning library
========================================================

Full documentation is available at: http://192.168.20.2:5700
"""

try:
    from mylearn.version import VERSION as __version__  # flake8: noqa
except ImportError:
    import sys
    print("ERROR: Version information not found.  Ensure you have built the "
          "software.\n    From the top level dir issue: 'make build'")
    sys.exit(1)
