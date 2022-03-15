"""
This module defines custom warnings and exceptions used across the Alibi Detect library.
"""
import functools
import warnings
from typing import Dict, Any, Callable


def deprecated_alias(**aliases: str) -> Callable:
    """
    Function decorator to warn about deprecated kwargs (and replace them).
    """
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            _rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)
        return wrapper
    return deco


def _rename_kwargs(func_name: str, kwargs: Dict[str, Any], aliases: Dict[str, str]):
    """
    Private function to rename deprecated kwarg to new name, and raise DeprecationWarning.
    """
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise ValueError(f"{func_name} received both the deprecated kwarg `{alias}` "
                                 f"and it's replacement `{new}`.")
            warnings.warn(f'`{alias}` is deprecated; use `{new}`.', UserWarning, stacklevel=3)
            kwargs[new] = kwargs.pop(alias)
