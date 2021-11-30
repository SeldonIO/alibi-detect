"""
Defining types compatible with different Python versions and defining custom types.
"""
import sys

if sys.version_info >= (3, 8):
    from typing import Literal  # noqa
else:
    from typing_extensions import Literal  # noqa
