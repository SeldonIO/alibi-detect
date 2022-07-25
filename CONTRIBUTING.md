# Contributing Code

We welcome PRs from the community. This document outlines the standard
practices and development tools we use.

When you contribute code, you affirm that the contribution is your original work and that you license the work to the project under the project's open source license. Whether or not you state this explicitly, by submitting any copyrighted material via pull request, email, or other means you agree to license the material under the project's open source license and warrant that you have the legal authority to do so.

## Getting started
The easiest way to get started is to install all the development dependencies
in a separate virtual environment:
```
pip install -r requirements/dev.txt -r requirements/docs.txt
```
This will install everything needed to run alibi-detect and all the dev tools
(docs builder, testing, linting etc.)

## Git pre-commit hooks
To make it easier to format code locally before submitting a PR, we provide
integration with `pre-commit` to run `flake8` and `mypy` hooks before every commit.
After installing the development requirements and cloning the package, run `pre-commit install`
from the project root to install the hooks locally. Now before every `git commit ...`
these hooks will be run to verify that the linting and type checking is correct. If there are
errors, the commit will fail and you will see the changes that need to be made.

## Testing
We use `pytest` to run tests. To run all tests just call `pytest alibi_detect` from the root of the project.
Test files live together with the library files under `tests` folders.

## Linter
We use `flake8` for linting adhering to PEP8 with exceptions defined in `setup.cfg`.

## Type checking
We use type hints to develop the libary and `mypy` to for static type checking. Some
options are defined in `setup.cfg`.

## Docstrings
We adhere to the `numpy` style docstrings (https://numpydoc.readthedocs.io/en/stable/format.html)
with the exception of ommiting argument types in docstrings in favour of type hints in function
and class signatures. If you're using a `PyCharm`, you can configure this under
`File -> Settings -> Tools -> Python Integrated Tools -> Docstrings`.

## Building documentation
We use `sphinx` for building documentation. You can call `make build_docs` from the project root,
the docs will be built under `doc/_build/html`.

## CI
All PRs triger a Github Actions  build to run linting, type checking, tests, and build docs.

## Optional Dependencies

Alibi-detect uses optional dependencies to allow users to avoid installing large or challenging to install dependencies. 
Alibi-detect manages modularity of components that depend on optional dependencies using the `import_optional` function 
defined in `alibi_detect/utils/missing_optional_dependency.py`. This replaces the dependency with a dummy object that 
raises an error when called. If you are working on public functionality that is dependent on an optional dependency you 
should expose the functionality via the relevant `__init__.py` file by importing it there using the `optional_import` 
function. Currently, optional dependencies are tested by importing all the public functionality and checking that the 
correct errors are raised dependent on the environment. Developers can run these tests using `tox`. These tests are in 
`alibi_detect/tests/test_dep_mangement.py`. If implementing functionality that is dependent on a new optional dependency
then you will need to:

1. Add it to `extras_require` in `setup.py`.
2. Create a new `tox` environment in `setup.cfg` with the new dependency.
3. Add a new dependency mapping to `ERROR_TYPES` in `alibi_detect/utils/missing_optional_dependency.py`. 
4. Make sure any public functionality is protected by the `import_optional` function.
5. Make sure the new dependency is tested in `alibi_detect/tests/test_dep_mangement.py`.

Note that subcomponents can be dependent on optional dependencies too. In this case the user should be able to import 
and use the relevant parent component. The user should only get an error message if: 
1. They don't have the optional dependency installed and,
2. They configure the parent component in such as way that it uses the subcomponent functionality.

Developers should implement this by importing the subcomponent into the source code defining the parent component using 
the `optional_import` function. The general layout of a subpackage with optional dependencies should look like: 

```
alibi_detect/subpackage/
  __init__.py  # expose public API with optional import guards
  defaults.py  # private implementations requiring only core deps
  optional_dep.py # private implementations requiring an optional dependency (or several?)
```

any public functionality that is dependent on an optional dependency should be imported into `__init__.py` using the 
`import_optional` function. 

#### Note:
- The `import_optional` function returns an instance of a class and if this is passed to type-checking constructs, such 
as Union, it will raise errors. Thus, in order to do type-checking, we need to 1. Conditionally import the true object 
dependent on `TYPE_CHECKING` and 2. Use forward referencing when passing to typing constructs such as `Union`. We use 
forward referencing because in a user environment the optional dependency may not be installed in which case it'll be 
replaced with an instance of the MissingDependency class. For example: 
  ```py
  from typing import TYPE_CHECKING, Union

  if TYPE_CHECKING:
    # Import for type checking. This will be type `UsefulClass`. Note import is from implementation file.
    from alibi_detect.utils.useful_classes import UsefulClass
  else:
    # Import is from `__init__` public API file. Class will be protected by optional_import function and so this will 
    # be type any.
    from alibi.utils import UsefulClass
  
  # The following will not throw an error because of the forward reference but mypy will still work.
  def example_function(useful_class: Union['UsefulClass', str]) -> None:
    ...
  ```
- Developers can use `make repl tox-env=<tox-env-name>` to run a python REPL with the specified optional dependency 
installed. This is to allow manual testing.