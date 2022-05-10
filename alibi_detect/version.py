try:
    from ._version import version  # noqa: F401
except ModuleNotFoundError:
    raise ImportError('alibi-detect must be installed before it can be imported.')
