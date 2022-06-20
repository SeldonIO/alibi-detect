"""Functionality for optional importing
This module provides a way to import optional dependencies. In the case that the user imports some functionality from
alibi that is not usable due to missing optional dependencies this code is used to allow the import but replace it
with an object that throws an error on use. This way we avoid errors at import time that prevent the user using
functionality independent of the missing dependency.
"""


from typing import Union, List, Optional, Any
from string import Template
from importlib import import_module

err_msg_template = Template((
    "Attempted to use $object_name without the correct optional dependencies installed. To install "
    + "the correct optional dependencies, run `pip install alibi-detect[$missing_dependency]` "
    + "from the command line. For more information, check the Installation documentation "
    + "at https://docs.seldon.io/projects/alibi-detect/en/latest/overview/getting_started.html."
))


ERROR_TYPES = {
    "fbprophet": 'prophet',
    "holidays": 'prophet',
    "pystan": 'prophet',
    "numba": 'numba',
    "tensorflow_probability": 'tensorflow_probability',
    "tensorflow": 'tensorflow',
    "torch": 'torch',
    "pytorch": 'torch'
}

ERROR_API_MAP = {
    "LLR": 'tensorflow,tensorflow_probability',
    "OutlierVAE": 'tensorflow,tensorflow_probability',
    "OutlierVAEGMM": 'tensorflow,tensorflow_probability',
    "OutlierAEGMM": 'tensorflow,tensorflow_probability',
    "OutlierSeq2Seq": 'tensorflow,tensorflow_probability',
    "AdversarialAE": 'tensorflow,tensorflow_probability',
    "ModelDistillation": 'tensorflow,tensorflow_probability',
    "PixelCNN": 'tensorflow,tensorflow_probability'
}


class MissingDependency:
    """Missing Dependency Class
    Used to replace any object that requires unmet optional dependencies. Attribute access or calling the __call__
    method on this object will raise an error.
    """
    def __init__(self,
                 object_name: str,
                 err: Union[ModuleNotFoundError, ImportError],
                 missing_dependency: str = 'all',):
        """ Metaclass for MissingDependency classes
        Parameters
        ----------
        object_name
            Name of object we are replacing
        missing_dependency
            Name of missing dependency required for object
        err
            Error to be raised when the class is initialized or used
        """
        self.missing_dependency = missing_dependency
        self.object_name = object_name
        self.err = err

    @property
    def err_msg(self):
        """Generate error message informing user to install missing dependencies."""
        return err_msg_template.substitute(
            object_name=self.object_name,
            missing_dependency=self.missing_dependency)

    def __getattr__(self, key):
        """Raise an error when attributes are accessed."""
        raise ImportError(self.err_msg) from self.err

    def __call__(self, *args, **kwargs):
        """If called, raise an error."""
        raise ImportError(self.err_msg) from self.err


def import_optional(module_name: str, names: Optional[List[str]] = None) -> Any:
    """Import a module that depends on optional dependencies
    Note: This function is used to import modules that depend on optional dependencies. Because it mirrors the python
    import functionality its return type has to be `Any`. Using objects imported with this function can lead to
    misspecification of types as `Any` when the developer intended to be more restrictive.

    Parameters
    ----------
        module_name
            The module to import
        names
            The names to import from the module. If None, all names are imported.
    Returns
    -------
        The module or named objects within the modules if names is not None. If the import fails due to a
        ModuleNotFoundError or ImportError then the requested module or named objects are replaced with instances of
        the MissingDependency class above.
    """
    if not names:
        names = []

    try:
        module = import_module(module_name)
        # TODO: We should check against specific dependency versions here.
        if names:
            objs = tuple(getattr(module, name) for name in names)
            return objs if len(objs) > 1 else objs[0]
        return module
    except (ImportError, ModuleNotFoundError) as err:
        if err.name is None:
            raise TypeError()
        if str(err.name) not in ERROR_TYPES:
            raise err

        # The object being imported might have multiple optional dependencies. Such cases are captured in ERROR_API_MAP.
        # We first check for these and assign to missing_dependency, if this isn't the case we etect hte optional
        # dependency option using ERROR_TYPES.
        missing_dependency = None
        for name in names:
            missing_dependency = ERROR_API_MAP.get(name)
        if not missing_dependency:
            missing_dependency = ERROR_TYPES[err.name]

        if names:
            missing_dependencies = \
                tuple(MissingDependency(
                    missing_dependency=missing_dependency,
                    object_name=name,
                    err=err) for name in names)
            return missing_dependencies if len(missing_dependencies) > 1 else missing_dependencies[0]
        return MissingDependency(
            missing_dependency=missing_dependency,
            object_name=module_name,
            err=err)