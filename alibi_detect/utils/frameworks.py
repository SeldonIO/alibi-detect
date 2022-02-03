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
