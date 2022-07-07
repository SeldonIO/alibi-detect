from .missing_optional_dependency import ERROR_TYPES
from typing import Optional, List, Dict

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


HAS_BACKEND = {
    'tensorflow': has_tensorflow,
    'sklearn': has_sklearn,
    'pytorch': has_pytorch,
}


def _iter_to_str(iterable):
    """ Correctly format iterable of items to comma seperated sentence string."""
    items = [f'`{option}`' for option in iterable]
    last_item_str = f'{items[-1]}' if not items[:-1] else f' and {items[-1]}'
    return ', '.join(items[:-1]) + last_item_str


class BackendValidator:
    def __init__(self, backend_options: Dict[Optional[str], List[str]], construct_name: str):
        """Checks for requires sets of backend options.

        Takes a dictionary of backends plus extra dependencies and generates correct error messages if they are unmet.

        Parameters
        ----------
        backend_options
            Dictionary from backend to list of dependencies that must be satisfied. An example of `backend_options`
            would be `{'tensorflow': ['tensorflow', 'tensorflow_probability'], 'pytorch': ['pytorch'], None: []}`. This
            would mean `'tensorflow'`, `'pytorch'` or `None` are available backend options. If the user passes a
            different backend they will receive and error listing the correct backends. In addition, the tensorflow
            backend requires both `tensorflow` and `tensorflow_probability` dependencies to be met. If one of these are
            missing the validator will issue an error message telling the user what dependencies to install.
        construct_name
            Name of the object that has a set of backends we need to verify.
        """
        self.backend_options = backend_options
        self.construct_name = construct_name

    def verify_backend(self, backend: str):
        """Verifies backend choice.

        Verifies backend is implemented and that the correct dependencies are installed for the requested backend. If
        the backend is not implemented or a dependency is missing then an error is issued.

        Parameters
        ----------
        backend
            Choice of backend the user wishes to initialize the alibi-detect construct with. Must be one of the keys
            in the `self.backend_options` dictionary.

        Raises
        ------
        NotImplementedError
            If backend is not a member of `self.backend_options.keys()` a `NotImplementedError` is raised. Note `None`
            is a valid choice of backend if it is set as a key on `self.backend_options.keys()`. If a backend is not
            implemented for an alibi-detect object then it should not have a key on `self.backend_options`.
        ImportError
            If one of the dependencies in `self.backend_options[backend]` is missing then an ImportError will be thrown
            including a message informing the user how to install.
        """
        if backend not in self.backend_options:
            self._raise_implementation_error(backend)

        dependencies = self.backend_options[backend]
        missing_deps = []
        for dependency in dependencies:
            if not HAS_BACKEND[dependency]:
                missing_deps.append(dependency)

        if missing_deps:
            self._raise_import_error(missing_deps, backend)

    def _raise_import_error(self, missing_deps: List[str], backend: str):
        """Raises import error if backend choice has missing dependency."""

        optional_dependencies = set(ERROR_TYPES[missing_dep] for missing_dep in missing_deps)
        missing_deps_str = _iter_to_str(missing_deps)
        error_msg = (f'{missing_deps_str} not installed. Cannot initialize and run {self.construct_name} '
                     f'with {backend} backend.')
        pip_msg = '' if not optional_dependencies else \
            (f'The necessary missing dependencies can be installed using '
             f'`pip install alibi-detect[{" ".join(optional_dependencies)}]`.')
        raise ImportError(f'{error_msg} {pip_msg}')

    def _raise_implementation_error(self, backend: str):
        """Raises NotImplementedError error if backend choice is not implemented."""

        backend_list = _iter_to_str(self.backend_options.keys())
        raise NotImplementedError(f"{backend} backend not implemented. Use one of {backend_list} instead.")
