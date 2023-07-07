# Change Log

## [v0.11.4](https://github.com/SeldonIO/alibi-detect/tree/v0.11.4) (2023-07-07)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.11.3...v0.11.4)

This is a patch release to support `numpy >= 1.24`, and drop official support for Python 3.7.

### Fixed
- Replace numpy aliases of builtin types like `np.int`, in order to support `numpy >= 1.24` ([#826](https://github.com/SeldonIO/alibi-detect/pull/826)).

### Development
- Drop official support for Python 3.7 ([#825](https://github.com/SeldonIO/alibi-detect/pull/825)).


## [v0.11.3](https://github.com/SeldonIO/alibi-detect/tree/v0.11.3) (2023-06-21)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.11.2...v0.11.3)

This is a patch release to officially enable support for Python 3.11.<br>
This is the last release with official support for Python 3.7.

### Development
- Test library on Python 3.11 ([#817](https://github.com/SeldonIO/alibi-detect/pull/817)).
- Separate code quality into its own Github Action and only run against the main development version of Python, currently Python 3.10 ([#793](https://github.com/SeldonIO/alibi-detect/pull/793)).
- Check and remove stale `mypy` ignore commands ([#794](https://github.com/SeldonIO/alibi-detect/pull/794)).
- Add developer instructions for docstring formatting ([#789](https://github.com/SeldonIO/alibi-detect/pull/789)).
- Bump `scikit-image` version to `0.21.x` ([#803](https://github.com/SeldonIO/alibi-detect/pull/803)).
- Bump `numba` version to `0.57.x` ([#783](https://github.com/SeldonIO/alibi-detect/pull/783)).
- Bump `sphinx` version to `7.x` ([#782](https://github.com/SeldonIO/alibi-detect/pull/782)).


## [v0.11.2](https://github.com/SeldonIO/alibi-detect/tree/v0.11.2) (2023-04-28)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.11.1...v0.11.2)

### Fixed
- Failure of `plot_feature_outlier_image` utility function when no outliers are detected ([#774](https://github.com/SeldonIO/alibi-detect/pull/774) - thanks [@signupatgmx](https://github.com/signupatgmx) !).

### Changed
 - Refactored methods that use `tensorflow` optimizers to work with the new optimizers introduced in `2.11` ([#739](https://github.com/SeldonIO/alibi-detect/pull/739)).
 - Maximum supported version of `tensorflow` bumped to `2.12.x` ([#764](https://github.com/SeldonIO/alibi-detect/pull/764)).
 - Maximum supported version of `tensorflow-probability` version to `0.19.x` ([#687](https://github.com/SeldonIO/alibi-detect/pull/687)).
 - Supported version of `pandas` bumped to `>1.0.0, <3.0.0` ([#765](https://github.com/SeldonIO/alibi-detect/pull/765)).
 - Maximum supported version of `scikit-image` bumped to `0.20.x` ([#751](https://github.com/SeldonIO/alibi-detect/pull/751)).

### Development
- Migrate `codecov` to use Github Actions and don't fail CI on coverage report upload failure due to rate limiting ([#768](https://github.com/SeldonIO/alibi-detect/pull/768), [#776](https://github.com/SeldonIO/alibi-detect/pull/776)).
 - Bump `mypy` version to `>=1.0, <2.0` ([#754](https://github.com/SeldonIO/alibi-detect/pull/754)). 
 - Bump `sphinx` version to `6.x` ([#709](https://github.com/SeldonIO/alibi-detect/pull/709)).
 - Bump `sphinx-design` version to `0.4.1` ([#769](https://github.com/SeldonIO/alibi-detect/pull/769)).
 - Bump `nbsphinx` version to `0.9.x` ([#757](https://github.com/SeldonIO/alibi-detect/pull/757)).
 - Bump `myst-parser` version to `>=1.0, <2.0` ([#756](https://github.com/SeldonIO/alibi-detect/pull/756)).
 - Bump `twine` version to `4.x` ([#511](https://github.com/SeldonIO/alibi-detect/pull/511)).
 - Bump `pre-commit` version to `3.x` and update the config ([#731](https://github.com/SeldonIO/alibi-detect/pull/731)).

## v0.11.1
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.11.0...v0.11.1)

### Fixed

- Fixed two bugs with the saving/loading of drift detector `preprocess_fn`'s ([#752](https://github.com/SeldonIO/alibi-detect/pull/752)):
  - When `preprocess_fn` was a custom Python function wrapped in a partial, kwarg's were not serialized. This has now been fixed.
  - When saving drift detector `preprocess_fn`'s, the filenames for kwargs saved to `.dill` files are now prepended with the kwarg name. This avoids files being overwritten if multiple kwargs are saved to `.dill`.

## v0.11.0
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.10.5...v0.11.0)

### Added
- **New feature** The [MMD](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/mmddrift.html) and [learned-kernel MMD](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/learnedkerneldrift.html) drift detectors have been extended with [KeOps](https://www.kernel-operations.io/keops/index.html) backends to scale and speed up the detectors. See the [example notebook](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_mmd_keops.html) for more info ([#548](https://github.com/SeldonIO/alibi-detect/pull/548) and [#602](https://github.com/SeldonIO/alibi-detect/pull/602)).
- **New feature** Added support for serializing detectors with PyTorch backends, and detectors containing PyTorch models in their proprocessing functions ([#656](https://github.com/SeldonIO/alibi-detect/pull/656)).
- **New feature** Added support for serializing detectors with scikit-learn and KeOps backends ([#642](https://github.com/SeldonIO/alibi-detect/pull/642) and [#681](https://github.com/SeldonIO/alibi-detect/pull/681)).
- **New feature** Added support for saving and loading online detectors' state. This allows a detector to be restarted from previously generated checkpoints ([#604](https://github.com/SeldonIO/alibi-detect/pull/604)).
- **New feature** Added a PyTorch version of the `UAE` preprocessing utility function ([#656](https://github.com/SeldonIO/alibi-detect/pull/656), ([#705](https://github.com/SeldonIO/alibi-detect/pull/705)).
- For the `ClassifierDrift` and `SpotTheDiffDrift` detectors, we can also return the out-of-fold instances of the reference and test sets. When using `train_size` for training the detector, this allows to associate the returned prediction probabilities with the correct instances ([#665](https://github.com/SeldonIO/alibi-detect/pull/665)).

### Changed
- Minimum `prophet` version bumped to `1.1.0` (used by `OutlierProphet`). This upgrade removes the dependency on `pystan` as `cmdstanpy` is used instead. This version also comes with pre-built wheels for all major platforms and Python versions, making both installation and testing easier ([#627](https://github.com/SeldonIO/alibi-detect/pull/627)).
- **Breaking change** The configuration field `config_spec` has been removed. In order to load detectors serialized from previous Alibi Detect versions, the field will need to be deleted from the detector's `config.toml` file. However, in any case, serialization compatibility across Alibi Detect versions is not currently guranteed. ([#641](https://github.com/SeldonIO/alibi-detect/pull/641)).
- Added support for serializing tensorflow optimizers. Previously, tensorflow optimizers were not serialized, which meant the default `optimizer` kwarg would also be set when a detector was loaded with `load_detector`, regardless of the `optimizer` given to the original detector ([#656](https://github.com/SeldonIO/alibi-detect/pull/656)).
- Strengthened pydantic validation of detector configs. The `flavour` backend is now validated whilst taking into account the optional dependencies. For example, a `ValidationError` will be raised if `flavour='pytorch'` is given but PyTorch is not installed ([#656](https://github.com/SeldonIO/alibi-detect/pull/656)).
- If a `categories_per_feature` dictionary is not passed to `TabularDrift`, a warning is now raised to inform the user that all features are assumed to be numerical ([#606](https://github.com/SeldonIO/alibi-detect/pull/606)).
- For better clarity, the original error is now reraised when optional dependency errors are raised ([#783](https://github.com/SeldonIO/alibi-detect/pull/783)).
- The maximum `tensorflow` version has been bumped from 2.9 to 2.10 ([#608](https://github.com/SeldonIO/alibi-detect/pull/608)).
- The maximum `torch` version has been bumped from 1.12 to 1.13 ([#669](https://github.com/SeldonIO/alibi-detect/pull/669)).

### Fixed
- Fixed an issue with the serialization of `kernel_a` and `kernel_b` in `DeepKernel`'s ([#656](https://github.com/SeldonIO/alibi-detect/pull/656)).
- Fixed minor documentation issues ([#636](https://github.com/SeldonIO/alibi-detect/pull/636), [#640](https://github.com/SeldonIO/alibi-detect/pull/640), [#651](https://github.com/SeldonIO/alibi-detect/pull/651)).
- Fixed an issue with a warning being incorrectly raised when `device='cpu'` was passed to PyTorch based detectors ([#698](https://github.com/SeldonIO/alibi-detect/pull/698)).
- Fixed a bug that could cause `IndexError`'s to be raised in the TensorFlow `MMDDriftOnline` detector when older `numpy` versions were installed ([#710](https://github.com/SeldonIO/alibi-detect/pull/710)).

### Development
- UTF-8 decoding is enforced when `README.md` is opened by `setup.py`. This is to prevent pip install errors on systems with `PYTHONIOENCODING` set to use other encoders ([#605](https://github.com/SeldonIO/alibi-detect/pull/605)).
- Skip specific save/load tests that require downloading remote artefacts if the relevant URI(s) is/are down ([#607](https://github.com/SeldonIO/alibi-detect/pull/607)).
- CI `test/` directories are now ignored when measuring testing code coverage. This has a side-effect of lowering the reported test coverage ([#614](https://github.com/SeldonIO/alibi-detect/pull/614)).
- Added codecov tags to measure to platform-specific code coverage ([#615](https://github.com/SeldonIO/alibi-detect/pull/615)).
- Added option to ssh into CI runs for debugging ([#644](https://github.com/SeldonIO/alibi-detect/pull/644)).
- Measure executation time of test runs in CI ([#712](https://github.com/SeldonIO/alibi-detect/pull/712)).

## v0.10.5
## [v0.10.5](https://github.com/SeldonIO/alibi-detect/tree/v0.10.5) (2023-01-26)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.10.4...v0.10.5)

### Fixed
- Fixed two bugs preventing backward compatibility when loading detectors saved with `<v0.10.0`
([#729](https://github.com/SeldonIO/alibi-detect/pull/729) and [#732](https://github.com/SeldonIO/alibi-detect/pull/732)). This bug also meant that detectors
saved with `save_detector(..., legacy=True)` in `>=v0.10.0` did not properly obey the legacy file format. The `config.toml` file format used by default in `>=v0.10.0` is unaffected. 

## v0.10.4
## [v0.10.4](https://github.com/SeldonIO/alibi-detect/tree/v0.10.4) (2022-10-21)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.10.3...v0.10.4)

### Fixed
- Fixed an incorrect default value for the `alternative` kwarg in the `FETDrift` detector ([#661](https://github.com/SeldonIO/alibi-detect/pull/661)).
- Fixed an issue with `ClassifierDrift` returning incorrect prediction probabilities when `train_size` given ([#662](https://github.com/SeldonIO/alibi-detect/pull/662)).

## [v0.10.3](https://github.com/SeldonIO/alibi-detect/tree/v0.10.3) (2022-08-17)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.10.2...v0.10.3)

### Fixed
- Fix to allow `config.toml` files to be loaded when the [meta] field is not present ([#591](https://github.com/SeldonIO/alibi-detect/pull/591)).

## [v0.10.2](https://github.com/SeldonIO/alibi-detect/tree/v0.10.2) (2022-08-16)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.10.1...v0.10.2)

### Fixed
- Fixed a bug in the MMDDrift detector with `pytorch` backend, where the `kernel` attribute was not sent to the selected device ([#587](https://github.com/SeldonIO/alibi-detect/pull/587)).

### Development
- Code Coverage added ([#584](https://github.com/SeldonIO/alibi-detect/pull/584)).

## [v0.10.1](https://github.com/SeldonIO/alibi-detect/tree/v0.10.1) (2022-08-10)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.10.0...v0.10.1)

### Fixed
- Corrected a missing optional dependency error when `tensorflow` was installed without `tensorflow-probability` ([#580](https://github.com/SeldonIO/alibi-detect/pull/580)).

### Development
- An upper version bound has been added for `torch` (<1.13.0) ([#575](https://github.com/SeldonIO/alibi-detect/pull/575)).

## [v0.10.0](https://github.com/SeldonIO/alibi-detect/tree/v0.10.0) (2022-07-26)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.9.1...v0.10.0)

### Added
- **New feature** Drift detectors save/load functionality has been significantly reworked. All offline and online drift detectors (`tensorflow` backend only) can now be saved and loaded via `config.toml` files, allowing for more flexibility. Config files are also validated with `pydantic`. See [the documentation](https://docs.seldon.io/projects/alibi-detect/en/stable/overview/config_files.html) for more info ([#516](https://github.com/SeldonIO/alibi-detect/pull/516)).
- **New feature** Option to use out-of-bag predictions when using a `RandomForestClassifier` with `ClassifierDrift` ([#426](https://github.com/SeldonIO/alibi-detect/pull/426)).
- Python 3.10 support. Note that PyTorch at the time of writing doesn't support Python 3.10 on Windows ([#485](https://github.com/SeldonIO/alibi-detect/pull/485)).

### Fixed
- Fixed a bug in the TensorFlow trainer which occured when the data was a minibatch of size 2 ([#492](https://github.com/SeldonIO/alibi-detect/pull/492)).

### Changed
- TensorFlow is now an optional dependency. Error messages for incorrect use of detectors that are dependent on missing optional dependencies have been improved to include installation instructions and be more informative ([#537](https://github.com/SeldonIO/alibi-detect/pull/537)).
- The optional dependency work has resulted in some imports being reorganised. The original imports will still work as long as the relevant optional dependencies are installed ([#538](https://github.com/SeldonIO/alibi-detect/pull/538)).
  - `from alibi_detect.utils.tensorflow.kernels import DeepKernel` -> `from alibi_detect.utils.tensorflow import DeepKernel`
  - `from alibi_detect.utils.tensorflow.prediction import predict_batch` -> `from alibi_detect.utils.tensorflow import predict_batch`
  - `from alibi_detect.utils.pytorch.data import TorchDataset` -> `from alibi_detect.utils.pytorch import TorchDataset`
  - `from alibi_detect.models.pytorch.trainer import trainer` -> `from alibi_detect.models.pytorch import trainer`
  - `from alibi_detect.models.tensorflow.resnet import scale_by_instance` -> `from alibi_detect.models.tensorflow import scale_by_instance`
  - `from alibi_detect.models.tensorflow.resnet import scale_by_instance` -> `from alibi_detect.models.tensorflow import scale_by_instance`
  - `from alibi_detect.utils.pytorch.kernels import DeepKernel` -> `from alibi_detect.utils.pytorch import DeepKernel`
  - `from alibi_detect.models.tensorflow.autoencoder import eucl_cosim_features` -> `from alibi_detect.models.tensorflow import eucl_cosim_features`
  - `from alibi_detect.utils.tensorflow.prediction import predict_batch` -> `from alibi_detect.utils.tensorflow import predict_batch`
  - `from alibi_detect.models.tensorflow.losses import elbo` -> `from alibi_detect.models.tensorflow import elbo`
  - `from alibi_detect.models import PixelCNN` -> `from alibi_detect.models.tensorflow import PixelCNN`
  - `from alibi_detect.utils.tensorflow.data import TFDataset` -> `from alibi_detect.utils.tensorflow import TFDataset`
  - `from alibi_detect.utils.pytorch.data import TorchDataset` -> `from alibi_detect.utils.pytorch import TorchDataset`
- The maximum `tensorflow` version has been bumped from 2.8 to 2.9 ([#508](https://github.com/SeldonIO/alibi-detect/pull/508)).
- **breaking change** The `detector_type` field in the `detector.meta` dictionary now indicates whether a detector is a 'drift', 'outlier' or 'adversarial' detector. Its previous meaning, whether a detector is online or offline, is now covered by the `online` field ([#564](https://github.com/SeldonIO/alibi-detect/pull/564)).

### Development
- Added `MissingDependency` class and `import_optional` for protecting objects that are dependent on optional dependencies ([#537](https://github.com/SeldonIO/alibi-detect/pull/537)).
- Added `BackendValidator` to factor out similar logic across detectors with backends ([#538](https://github.com/SeldonIO/alibi-detect/pull/538)).
- Added missing CI test for `ClassifierDrift` with `sklearn` backend ([#523](https://github.com/SeldonIO/alibi-detect/pull/523)).
- Fixed typing for `ContextMMDDrift` `pytorch` backend with `numpy`>=1.22 ([#520](https://github.com/SeldonIO/alibi-detect/pull/520)).
- Drift detectors with backends refactored to perform distance threshold computation in `score` instead of `predict` ([#489](https://github.com/SeldonIO/alibi-detect/pull/489)).
- Factored out PyTorch device setting to `utils.pytorch.misc.get_device()` ([#503](https://github.com/SeldonIO/alibi-detect/pull/503)). Thanks to @kuutsav!
- Added `utils._random` submodule and `pytest-randomly` to manage determinism in CI build tests ([#496](https://github.com/SeldonIO/alibi-detect/pull/496)).
- From this release onwards we exclude the directories `doc/` and `examples/` from the source distribution (by adding `prune` directives in `MANIFEST.in`). This results in considerably smaller file sizes for the source distribution.
- `mypy` has been updated to `~=0.900` which requires additional development dependencies for type stubs, currently only `types-requests` and `types-toml` have been necessary to add to `requirements/dev.txt`.

## [v0.9.1](https://github.com/SeldonIO/alibi-detect/tree/v0.9.1) (2022-04-01)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.9.0...v0.9.1)

### Fixed
- Fixed an issue whereby simply importing the library in any capacity caused tensorflow to occupy all available GPU memory. This was due to the instantiation of `tf.keras.Model` objects within a class definition (`GaussianRBF` objects within the `DeepKernel` class).

## [v0.9.0](https://github.com/SeldonIO/alibi-detect/tree/v0.9.0) (2022-03-17)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.8.1...v0.9.0)

### Added
- Added the [ContextMMDDrift](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/contextmmddrift.html) detector. The context-aware maximum mean discrepancy drift detector ([Cobb and Van Looveren, 2022](https://arxiv.org/abs/2203.08644)) is a kernel based method for detecting drift in a manner that can take relevant context into account.

### Fixed
- Fixed an issue experienced when the [Model uncertainty based drift detection](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_model_unc_cifar10_wine.html) example is run on GPU's ([#445](https://github.com/SeldonIO/alibi-detect/pull/445)).
- Fixed an issue with the [Text drift detection on IMDB](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_text_imdb.html) example to allow PyTorch to be used ([#438](https://github.com/SeldonIO/alibi-detect/pull/438)).

### Development
- The maximum `tensorflow` version has been bumped from 2.7 to 2.8 ([#444](https://github.com/SeldonIO/alibi-detect/pull/444)).

## [v0.8.1](https://github.com/SeldonIO/alibi-detect/tree/v0.8.1) (2022-01-18)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.8.0...v0.8.1)

### Added
- **New feature** `ClassifierDrift` now supports `sklearn` models ([#414](https://github.com/SeldonIO/alibi-detect/pull/414)). See [this example](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_clf_adult.html). 

### Changed
- Python 3.6 has been deprecated from the supported versions as it has reached end-of-life. 

### Fixed
- The `SpectralResidual` detector now uses padding to prevent spikes occuring at the beginning and end of scores ([#396](https://github.com/SeldonIO/alibi-detect/pull/396)).
- The handling of url's in the dataset and model fetching methods has been modified to fix behaviour on Windows platforms.  

### Development
- `numpy` typing has been updated to be compatible with `numpy 1.22` ([#403](https://github.com/SeldonIO/alibi-detect/pull/403)). This is a prerequisite for upgrading to `tensorflow 2.7`. 
- The Alibi Detect CI tests now include Windows and MacOS platforms ([#423](https://github.com/SeldonIO/alibi-detect/pull/423)).
- The maximum `tensorflow` version has been bumped from 2.6 to 2.7 ([#377](https://github.com/SeldonIO/alibi-detect/pull/377)).

## [v0.8.0](https://github.com/SeldonIO/alibi-detect/tree/v0.8.0) (2021-12-09)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.7.3...v0.8.0)

### Added
- [Offline](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/fetdrift.html) and [online](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinefetdrift.html) versions of Fisher's Exact Test detector for supervised drift detection on binary data: `from alibi_detect.cd import FETDrift, FETDriftOnline`.
- [Offline](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/cvmdrift.html) and [online](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinecvmdrift.html) versions of Cramér-von Mises detector for supervised drift detection on continuous data: `from alibi_detect.cd import CVMDrift, CVMDriftOnline`.
- Offline supervised drift detection [example](https://github.com/SeldonIO/alibi-detect/blob/master/examples/cd_supervised_penguins.ipynb) on the penguin classification dataset.

### Changed
 - Refactored online detectors to separate updating of state ([#371](https://github.com/SeldonIO/alibi-detect/pull/371)). 
 - Update `tensorflow` lower bound to 2.2 due to minimum requirements from `transformers`. 

### Fixed
 - Fixed incorrect kwarg name in `utils.tensorflow.distance.permed_lsdd` function ([#399](https://github.com/SeldonIO/alibi-detect/pull/399)). 

### Development
 - Updated `sphinx` for documentation building to `>=4.2.0`.
 - Added a `CITATIONS.cff` file for consistent citing of the library.
 - CI actions are now not triggered on draft PRs (apart from a `readthedoc` build).
 - Removed dependency on `nbsphinx_link` and moved examples under `doc/source/examples` with symlinks from the top-level `examples` directory.

## [v0.7.3](https://github.com/SeldonIO/alibi-detect/tree/v0.7.3) (2021-10-29)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.7.2...v0.7.3)

### Added
- `DeepKernel` is allowed without the `kernel_b` component, giving a kernel consisting of only a deep kernel component (`kernel_a`). 
- Documentation layout refreshed, and a new "Background to drift detection" added.

### Fixed
- Model fetching methods now correctly handle nested filepaths.
- For backward compatibility, fetch and load methods now attept to fetch/load `dill` files, but fall back to `pickle` files. 
- Prevent `dill` from extending `pickle` dispatch table. This prevents undesirable behaviour if using `pickle`/`joblib` without `dill` imported later on (see #326).
- For consistency between `save_detector` and `load_detector`, `fetch_detector` will no longer append `detector_name` to `filepath`.

## [v0.7.2](https://github.com/SeldonIO/alibi-detect/tree/v0.7.2) (2021-08-17)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.7.1...v0.7.2)

### Added
- Learned kernels drift detector with TensorFlow and PyTorch support: `from alibi_detect.cd import LearnedKernelDrift`
- Spot-the-diff drift detector with TensorFlow and PyTorch support: `from alibi_detect.cd import SpotTheDiffDrift`
- Online drift detection example on medical imaging data: `https://github.com/SeldonIO/alibi-detect/blob/master/examples/cd_online_camelyon.ipynb`

## [v0.7.1](https://github.com/SeldonIO/alibi-detect/tree/v0.7.1) (2021-07-22)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.7.0...v0.7.1)

### Added
- Extend allowed input type for drift detectors to include List[Any] with additional graph and text data examples.
- Allow custom preprocessing steps within `alibi_detect.utils.pytorch.prediction.predict_batch` and `alibi_detect.utils.tensorflow.prediction.predict_batch`. This makes it possible to take List[Any] as input and combine instances in the list into batches of data in the right format for the model.

### Removed
- PCA preprocessing step for drift detectors.

### Fixed
- Improve numerical stability LSDD detectors (offline and online) to avoid overflow/underflow caused by higher dimensionality of the input data.
- Spectral Residual outlier detector test.

## [v0.7.0](https://github.com/SeldonIO/alibi-detect/tree/v0.7.0) (2021-06-07)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.6.2...v0.7.0)

### Added
- Least squares density difference drift detector `from alibi_detect.cd import LSDDDrift` with TensorFlow and PyTorch support.
- Online versions of the MMD and LSDD drift detectors: `from alibi_detect.cd import MMDDriftOnline, LSDDDriftOnline` with TensorFlow and PyTorch support.
- Enable Python 3.9 support.

### Fixed
- Hidden layer output as preprocessing step for drift detectors for internal layers with higher dimensional shape, e.g. `(B, C, H, W)`.

## [v0.6.2](https://github.com/SeldonIO/alibi-detect/tree/v0.6.2) (2021-05-06)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.6.1...v0.6.2)

### Fixed
- alibi-detect compatibility with transformers>=4.0.0
- update slack link to point to alibi-detect channel

## [v0.6.1](https://github.com/SeldonIO/alibi-detect/tree/v0.6.1) (2021-04-26)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.6.0...v0.6.1)

### Added
- Classification and regression model uncertainty drift detectors for both PyTorch and TensorFlow models: `from alibi_detect.cd import ClassifierUncertaintyDrift, RegressorUncertaintyDrift`.
- Return p-values for `ClassifierDrift` detectors using either a KS test on the classifier's probabilities or logits. The model predictions can also be binarised and a binomial test can be applied.
- Allow unseen categories in the test batches for the categorical and tabular drift detectors: `from alibi_detect.cd import ChiSquareDrift, TabularDrift`.


## [v0.6.0](https://github.com/SeldonIO/alibi-detect/tree/v0.6.0) (2021-04-12)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.5.1...v0.6.0)

### Added
- Flexible backend support (TensorFlow and PyTorch) for drift detectors `MMDDrift` and `ClassifierDrift` as well as support for both frameworks for preprocessing steps (`from alibi_detect.cd.tensorflow import HiddenOutput, preprocess_drift` and `from alibi_detect.models.tensorflow import TransformerEmbedding`, replace `tensorflow` with `pytorch` for PyTorch support) and various utility functions (kernels and distance metrics) under `alibi_detect.utils.tensorflow` and `alibi_detect.utils.pytorch`.
- Significantly faster implementation MMDDrift detector leveraging both GPU implementations in TensorFlow and PyTorch as well as making efficient use of the cached kernel matrix for the permutation tests.
- Change test for `ChiSquareDrift` from goodness-of-fit of the observed data against the empirical distribution of the reference data to a test for homogeneity which does not bias p-values as much to extremes.
- Include NumpyEncoder in library to facilitate json serialization.

### Removed
- As part of the introduction of flexible backends for various drift detectors, dask is no longer supported for the `MMDDrift` detector and distance computations.

### Fixed
- Update RTD theme version due to rendering bug.
- Bug when using `TabularDrift` with categorical features and continuous numerical features. Incorrect indexing of categorical columns was performed.

### Development
- Pin pystan version to working release with prophet.

## [v0.5.1](https://github.com/SeldonIO/alibi-detect/tree/v0.5.1) (2021-03-05)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.5.0...v0.5.1)

This is a bug fix release.

### Fixed
- The order of the reference and test dataset for the `TabularDrift` and `ChiSquareDrift` was reversed leading to incorrect test statistics
- The implementation of `TabularDrift` and `ChiSquareDrift` were not accounting for the different sample sizes between reference and test datasets leading to incorrect test statistics
- Bumped required `scipy` version to `1.3.0` as older versions were missing the `alternative` keyword argument for `ks_2samp` function 

## [v0.5.0](https://github.com/SeldonIO/alibi-detect/tree/v0.5.0) (2021-02-18)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.4.4...v0.5.0)
### Added
- Chi-square drift detector for categorical data: `alibi_detect.cd.chisquare.ChiSquareDrift`
- Mixed-type tabular data drift detector: `alibi_detect.cd.tabular.TabularDrift`
- Classifier-based drift detector: `alibi_detect.cd.classifier.ClassifierDrift`

### Removed
- DataTracker utility

### Development
- Docs build improvements, dependabot integration, daily build cronjob


## [v0.4.4](https://github.com/SeldonIO/alibi-detect/tree/v0.4.4) (2020-12-23)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.4.3...v0.4.4)
### Added
- Remove integrations directory
- Extend return dict drift detector
- Update saving functionality drift detectors

## [v0.4.3](https://github.com/SeldonIO/alibi-detect/tree/v0.4.3) (2020-10-08)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.4.2...v0.4.3)
### Added
- Make Prophet an optional dependency
- Extend what is returned by the drift detectors to raw scores
- Add licenses from dependencies

## [v0.4.2](https://github.com/SeldonIO/alibi-detect/tree/v0.4.2) (2020-09-09)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.4.1...v0.4.2)
### Added
- Text drift detector functionality for KS and MMD drift detectors
- Add embedding extraction functionality for pretrained HuggingFace transformers models (`alibi_detect.models.embedding`)
- Add Python 3.8 support

## [v0.4.1](https://github.com/SeldonIO/alibi-detect/tree/v0.4.1) (2020-05-12)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.4.0...v0.4.1)
### Added
- Likelihood ratio outlier detector (`alibi_detect.od.llr.LLR`) with image and genome dataset examples
- Add genome dataset (`alibi_detect.datasets.fetch_genome`)
- Add PixelCNN++ model (`alibi_detect.models.pixelcnn.PixelCNN`)

## [v0.4.0](https://github.com/SeldonIO/alibi-detect/tree/v0.4.0) (2020-04-02)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.3.1...v0.4.0)
### Added
- Kolmogorov-Smirnov drift detector (`alibi_detect.cd.ks.KSDrift`)
- Maximum Mean Discrepancy drift detector (`alibi_detect.cd.mmd.MMDDrift`)

## [v0.3.1](https://github.com/SeldonIO/alibi-detect/tree/v0.3.1) (2020-02-26)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.3.0...v0.3.1)
### Added
- Adversarial autoencoder detection method (offline method, `alibi_detect.ad.adversarialae.AdversarialAE`)
- Add pretrained adversarial and outlier detectors to Google Cloud Bucket and include fetch functionality
- Add data/concept drift dataset (CIFAR-10-C) to Google Cloud Bucket and include fetch functionality 
- Update VAE loss function and log var layer
- Fix tests for Prophet outlier detector on Python 3.6
- Add batch sizes for all detectors

## [v0.3.0](https://github.com/SeldonIO/alibi-detect/tree/v0.3.0) (2020-01-17)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.2.0...v0.3.0)
### Added
- Multivariate time series outlier detection method OutlierSeq2Seq (offline method, `alibi_detect.od.seq2seq.OutlierSeq2Seq`)
- ECG and synthetic data  examples for OutlierSeq2Seq detector
- Auto-Encoder outlier detector (offline method, `alibi_detect.od.ae.OutlierAE`)
- Including tabular and categorical perturbation functions (`alibi_detect.utils.perturbation`)

## [v0.2.0](https://github.com/SeldonIO/alibi-detect/tree/v0.2.0) (2019-12-06)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.1.0...v0.2.0)
### Added
 - Univariate time series outlier detection methods: Prophet (offline method, `alibi_detect.od.prophet.OutlierProphet`)
   and Spectral Residual (online method, `alibi_detect.od.sr.SpectralResidual`)
 - Function for fetching Numenta Anomaly Benchmark time series data (`alibi_detect.datasets.fetch_nab`)
 - Perturbation function for time series data (`alibi_detect.utils.perturbation.inject_outlier_ts`)
 - Roadmap

## [v0.1.0](https://github.com/SeldonIO/alibi-detect/tree/v0.1.0) (2019-11-19)
### Added
 - Isolation Forest (Outlier Detection)
 - Mahalanobis Distance (Outlier Detection)
 - Variational Auto-Encoder (VAE, Outlier Detection)
 - Auto-Encoding Gaussian Mixture Model (AEGMM, Outlier Detection)
 - Variational Auto-Encoding Gaussian Mixture Model (VAEGMM, Outlier Detection)
 - Adversarial Variational Auto-Encoder (Adversarial Detection)

