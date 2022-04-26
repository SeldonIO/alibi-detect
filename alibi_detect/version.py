# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module module
# The version is read from the pyproject.toml file
from importlib import metadata
__version__ = metadata.version(__package__)
