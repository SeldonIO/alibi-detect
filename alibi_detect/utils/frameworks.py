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


BACKENDS = ['tensorflow', 'pytorch']


def _validate_backend(backend):
    backend = backend.lower()
    if backend == 'tensorflow' and not has_tensorflow or backend == 'pytorch' and not has_pytorch:
        raise ImportError(f'{backend} not installed. ')
    elif backend not in BACKENDS:
        raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.')
