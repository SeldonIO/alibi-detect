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

def validate_backend(backend):
    BACKENDS = ['tensorflow', 'pytorch']
    backend = backend.lower() 
    if backend == 'tensorflow' and not has_tensorflow or backend == 'pytorch' and not has_pytorch: 
        raise ImportError(f'{backend} not installed. Cannot initialize and run the ' 
                        f'ClassifierDrift detector with {backend} backend.') 
    elif backend not in BACKENDS: 
        raise NotImplementedError(f'{backend} not implemented. Use tensorflow or pytorch instead.') 
