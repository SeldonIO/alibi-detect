try:
    import tensorflow as tf  # noqa
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

try:
    import torch  # noqa
    has_pytorch = True
except ImportError:
    has_pytorch = False

try:
    import sklearn  # noqa
    has_sklearn = True
except ImportError:
    has_sklearn = False

try:
    import pykeops  # noqa
    has_keops = True
except ImportError:
    has_keops = False
