# Change Log

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

