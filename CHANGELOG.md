# Change Log

## [v0.9.0](https://github.com/SeldonIO/alibi-detect/tree/v0.9.0) (2022-03-17)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.8.1...v0.9.0)

### Added
- Added the [ContextMMDDrift](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/contextmmddrift.html) detector. The context-aware maximum mean discrepancy drift detector ([Cobb and Van Looveren, 2022](https://arxiv.org/abs/2203.08644)) is a kernel based method for detecting drift in a manner that can take relevant context into account.
- The maximum `tensorflow` version has been bumped from 2.7 to 2.8 ([#444](https://github.com/SeldonIO/alibi-detect/pull/444)).

### Fixed
- Fixed an issue experienced when the [Model uncertainty based drift detection](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_model_unc_cifar10_wine.html) example is run on GPU's ([#445](https://github.com/SeldonIO/alibi-detect/pull/445)).
- Fixed an issue with the [Text drift detection on IMDB](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_text_imdb.html) example to allow PyTorch to be used ([#438](https://github.com/SeldonIO/alibi-detect/pull/438)).

## [v0.8.1](https://github.com/SeldonIO/alibi-detect/tree/v0.8.1) (2022-01-18)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.8.0...v0.8.1)

### Added
- **New feature** `ClassifierDrift` now supports `sklearn` models ([#414](https://github.com/SeldonIO/alibi-detect/pull/414)). See [this example](https://docs.seldon.io/projects/alibi-detect/en/latest/examples/cd_clf_adult.html). 
- The maximum `tensorflow` version has been bumped from 2.6 to 2.7 ([#377](https://github.com/SeldonIO/alibi-detect/pull/377)).

### Changed
- Python 3.6 has been deprecated from the supported versions as it has reached end-of-life. 

### Fixed
- The `SpectralResidual` detector now uses padding to prevent spikes occuring at the beginning and end of scores ([#396](https://github.com/SeldonIO/alibi-detect/pull/396)).
- The handling of url's in the dataset and model fetching methods has been modified to fix behaviour on Windows platforms.  

### Development
- `numpy` typing has been updated to be compatible with `numpy 1.22` ([#403](https://github.com/SeldonIO/alibi-detect/pull/403)). This is a prerequisite for upgrading to `tensorflow 2.7`. 
- The Alibi Detect CI tests now include Windows and MacOS platforms ([#423](https://github.com/SeldonIO/alibi-detect/pull/423)).

## [v0.8.0](https://github.com/SeldonIO/alibi-detect/tree/v0.8.0) (2021-12-09)
[Full Changelog](https://github.com/SeldonIO/alibi-detect/compare/v0.7.3...v0.8.0)

### Added
- [Offline](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/fetdrift.html) and [online](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/onlinefetdrift.html) versions of Fisher's Exact Test detector for supervised drift detection on binary data: `from alibi_detect.cd import FETDrift, FETDriftOnline`.
- [Offline](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/cvmdrift.html) and [online](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/onlinecvmdrift.html) versions of Cramér-von Mises detector for supervised drift detection on continuous data: `from alibi_detect.cd import CVMDrift, CVMDriftOnline`.
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

