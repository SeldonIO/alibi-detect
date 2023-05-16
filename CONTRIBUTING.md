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
and class signatures. If you use an IDE, you may be able to configure it to assist you with writing
docstrings in the correct format. For `PyCharm`, you can configure this under
`File -> Settings -> Tools -> Python Integrated Tools -> Docstrings`. For `Visual Studio Code`, you can obtain
docstring generator extensions from the [VisualStudio Marketplace](https://marketplace.visualstudio.com/).

When documenting Python classes, we adhere to the convention of including docstrings in their `__init__` method, 
rather than as a class level docstring. Docstrings should only be included at the class-level if a class does
not posess an `__init__` method, for example because it is a static class.

#### Conventions

- Names of variables, functions, classes and modules should be written between single back-ticks.
     - ``` A `numpy` scalar type that ```
     - ``` `X` ```
     - ``` `extrapolate_constant_perc` ```

- Simple mathematical equations should be written between single back-ticks to facilitate readability in the console.
     - ``` A callable that takes an `N x F` tensor, for ```
     - ``` `x >= v, fun(x) >= target` ```

- Complex math should be written in LaTeX.
    - ``` function where :math:`link(output - expected\_value) = sum(\phi)` ```

- Other `alibi_detect` objects should be cross-referenced using references of the form `` :role:`~object` ``, where 
`role` is one of the roles listed in the [sphinx documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects), 
and `object` is the full path of the object to reference. For example, the `MMDDrift` detectors's `predict` method 
would be referenced with `` :meth:`~alibi_detect.cd.mmd.MMDDrift.predict` ``. This will render as `MMDDrift.predict()` and 
link to the relevant API docs page. The same convention can be used to reference objects from other libraries, providing the
library is included in `intersphinx_mapping` in `doc/source/conf.py`. If the `~` is removed, the absolute object location will be
rendered.

- Variable values or examples of setting an argument to a specific values should be written in double back-ticks
to facilitate readability as they are rendered in a block with orange font-color.
   - ``` is set to ``True`` ```
   - ``` A list of features for which to plot the ALE curves or ``'all'`` for all features. ```
   - ``` The search is greedy if ``beam_size=1`` ```
   - ``` if the result uses ``segment_labels=(1, 2, 3)`` and ``partial_index=1``, this will return ``[1, 2]``. ```
   
- Listing the possible values an argument can take.
   - ``` Possible values are: ``'all'`` | ``'row'`` | ``None``. ```

- Returning the name of the variable and its description - standard convention and renders well. Writing the 
variable types should be avoided as it would be duplicated from variables typing.
```
Returns
-------
raw
    Array of perturbed text instances.
data
    Matrix with 1s and 0s indicating whether a word in the text has not been perturbed for each sample.
```

- Returning only the description. When the name of the variable is not returned, sphinx wrongly interprets the 
description as the variable name which will render the text in italic. If the text exceeds one line, ``` \ ``` need 
to be included after each line to avoid introducing bullet points at the beginning of each row. Moreover, if for 
example the name of a variable is included between single back-ticks, the italic font is canceled for all the words
with the exception of the ones inbetween single back-ticks.
```
Returns
-------
If the user has specified grouping, then the input object is subsampled and an object of the same \
type is returned. Otherwise, a `shap_utils.Data` object containing the result of a k-means algorithm \
is wrapped in a `shap_utils.DenseData` object and returned. The samples are weighted according to the \
frequency of the occurrence of the clusters in the original data.
```
 
- Returning an object which contains multiple attributes and each attribute is described individually. 
In this case the attribute name is written between single back-ticks and the type, if provided, would be written in 
double back-ticks.
```
Returns
-------
`Explanation` object containing the anchor explaining the instance with additional metadata as attributes. \
Contains the following data-related attributes

 - `anchor` : ``List[str]`` - a list of words in the proposed anchor.

 - `precision` : ``float`` - the fraction of times the sampled instances where the anchor holds yields \
 the same prediction as the original instance. The precision will always be  threshold for a valid anchor.

 - `coverage` : ``float`` - the fraction of sampled instances the anchor applies to.
```

- Documenting a dictionary follows the same principle the as above but the key should be written between 
double back-ticks.
```
Default perturbation options for ``'similarity'`` sampling

    - ``'sample_proba'`` : ``float`` - probability of a word to be masked.

    - ``'top_n'`` : ``int`` - number of similar words to sample for perturbations.

    - ``'temperature'`` : ``float`` - sample weight hyper-parameter if `use_proba=True`.

    - ``'use_proba'`` : ``bool`` - whether to sample according to the words similarity.
```

- Attributes are commented inline to avoid duplication.
```
class ReplayBuffer:
    """
    Circular experience replay buffer for `CounterfactualRL` (DDPG) ... in performance.
    """
    X: np.ndarray  #: Inputs buffer.
    Y_m: np.ndarray  #: Model's prediction buffer.
    ...
```

For more standard conventions, please check the [numpydocs style guide](https://numpydoc.readthedocs.io/en/stable/format.html).

## Building documentation
We use `sphinx` for building documentation. You can call `make build_docs` from the project root,
the docs will be built under `doc/_build/html`.

## CI
All PRs triger a Github Actions build to run linting, type checking, tests, and build docs. The status of each 
Github Action can be viewed on the [actions page](https://github.com/SeldonIO/alibi-detect/actions).

### Debugging via CI

For various reasons, CI runs might occasionally fail. They can often be debugged locally, but sometimes it is helpful
to debug them in the exact enviroment seen during CI. For this purpose, there is the facilty to ssh directly into
the CI Guthub Action runner.

#### Instructions

1. Go to the "CI" workflows section on the Alibi Detect GitHub Actions page.

2. Click on "Run Workflow", and select the "Enable tmate debugging" toggle.

3. Select the workflow once it starts, and then select the build of interest (e.g. `ubuntu-latest, 3.10`).

4. Once the workflow reaches the `Setup tmate session` step, click on the toggle to expand it.

5. Copy and paste the `ssh` command that is being printed to your terminal e.g. `ssh TaHAR65KFbjbSdrkwxNz996TL@nyc1.tmate.io`.

6. Run the ssh command locally. Assuming your ssh keys are properly set up for github, you should now be inside the GutHub Action runner.

7. The tmate session is opened after the Python and pip installs are completed, so you should be ready to run `alibi-detect` and debug as required. 

#### Additional notes 

- If the registered public SSH key is not your default private SSH key, you will need to specify the path manually, like so: ssh -i <path-to-key> <tmate-connection-string>.
- Once you have finished debugging, you can continue the workflow (i.e. let the full build CI run) by running `touch continue` whilst in the root directory (`~/work/alibi-detect/alibi-detect`). This will close the tmate session.
- This new capability is currently temperamental on the `MacOS` build due to [this issue](https://github.com/mxschmitt/action-tmate/issues/69). If the MacOS build fails all the builds are failed. If this happens, it is 
recommended to retrigger only the workflow build of interest e.g. `ubuntu-latest, 3.10`, and then follow the instructions above from step 3.

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
