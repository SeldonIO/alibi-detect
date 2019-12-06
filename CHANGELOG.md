# Change Log

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

