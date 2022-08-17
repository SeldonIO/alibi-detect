# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module module
__version__ = "0.10.3"

# Define the config specification version. This is distinct to the library version above. It is only updated when
# any detector config schema is updated, such that loading a previous config spec cannot be guaranteed to work.
# The minor version number is associated with minor changes such as adding/removing/changing kwarg's, whilst the major
# number is reserved for significant changes to the config layout.
__config_spec__ = "0.1"
