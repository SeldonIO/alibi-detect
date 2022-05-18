from .missing_optional_dependency import ERROR_TYPES

try:
    import tensorflow as tf  # noqa
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

try:
    import tensorflow_probability # noqa
    has_tensorflow_probability = True
except ImportError:
    has_tensorflow_probability = False

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


HAS_BACKEND = {
    'tensorflow': has_tensorflow,
    'tensorflow_probability': has_tensorflow_probability,
    'sklearn': has_sklearn,
    'pytorch': has_pytorch,
}


def iter_to_str(iterable):
    items = [f'`{option}`' for option in iterable]
    last_item_str = f'{items[-1]}' if not items[:-1] else f' and {items[-1]}'
    return ', '.join(items[:-1]) + last_item_str


class BackendValidator:
    def __init__(self, backend_options, construct_name):
        """Checks for requires sets of backend options

        Parameters
        ----------
        backend_options
            Dictionary from backend to list of dependencies that must be satisfied.
        construct_name
            Name of the object that has a set of backends we need to verify.
        """
        self.backend_options = backend_options
        self.construct_name = construct_name

    def verify_backend(self, backend):
        if backend not in self.backend_options:
            self.raise_implementation_error(backend)

        dependencies = self.backend_options[backend]
        missing_deps = []
        for dependency in dependencies:
            if not HAS_BACKEND[dependency]:
                missing_deps.append(dependency)

        if missing_deps:
            self.raise_import_error(missing_deps, backend)

    def raise_import_error(self, missing_deps, backend):
        optional_dependencies = set(ERROR_TYPES[missing_dep] for missing_dep in missing_deps)
        missing_deps_str = iter_to_str(missing_deps)
        error_msg = (f'{missing_deps_str} not installed. Cannot initialize and run {self.construct_name} '
                     f'with {backend} backend.')
        pip_msg = '' if not optional_dependencies else \
            (f'The necessary missing dependencies can be installed using '
             f'`pip install alibi-detect[{",".join(optional_dependencies)}]`.')
        raise ImportError(f'{error_msg} {pip_msg}')

    def raise_implementation_error(self, backend):
        backend_list = iter_to_str(self.backend_options.keys())
        raise NotImplementedError(f"{backend} backend not implemented. Use one of {backend_list} instead.")
