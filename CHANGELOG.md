# Change Log

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

