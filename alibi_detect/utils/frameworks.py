try:
    import tensorflow as tf
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

try:
    import torch
    has_pytorch = True
except ImportError:
    has_pytorch = False
