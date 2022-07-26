# Saving and Loading

Alibi Detect includes support for saving and loading detectors to disk. To 
save a detector, simply call the `save_detector` method and provide a path to a directory (a new
one will be created if it doesn't exist):

```python
from alibi_detect.od import OutlierVAE
from alibi_detect.saving import save_detector

od = OutlierVAE(...) 

filepath = './my_detector/'
save_detector(od, filepath)
```

To load a previously saved detector, use the `load_detector` method and provide it with the path to the detector's 
directory:

```python
from alibi_detect.saving import load_detector

filepath = './my_detector/'
od = load_detector(filepath)
```

```{warning}
When loading a saved detector, a warning will be issued if the runtime alibi-detect version is 
different from the version used to save the detector. **It is highly recommended to use the same 
alibi-detect, Python and dependency versions as were used to save the detector to avoid potential 
bugs and incompatibilities**.
```

## Formats
Detectors can be saved using two formats:

- **Config format**: For drift detectors, by default `save_detector` serializes the detector via a config file named `config.toml`, 
stored in `filepath`. The [TOML](https://toml.io/en/) format is human-readable, which makes the config files useful for 
record keeping, and allows a detector to be edited before it is reloaded. For more details, see 
[Detector Configuration Files](config_files.md).

- **Legacy format**: Outlier and adversarial detectors are saved to [dill](https://dill.readthedocs.io/en/latest/dill.html) files stored
within `filepath`. Drift detectors can also be saved in this legacy format by running `save_detector` with 
`legacy=True`. Loading is performed in the same way, by simply running `load_detector(filepath)`.


## Supported detectors

The following tables list the current state of save/load support for each detector. Adding full support 
for the remaining detectors is in the [Roadmap](roadmap.md).


````{tab-set}

```{tab-item} Drift detectors
| Detector                                                                       | Legacy save/load | Config save/load |
|:-------------------------------------------------------------------------------|:----------------:|:----------------:|
| [Kolmogorov-Smirnov](../cd/methods/ksdrift.ipynb)                              |        ✅         |        ✅         |
| [Cramér-von Mises](../cd/methods/cvmdrift.ipynb)                               |        ✅         |        ✅         |
| [Fisher's Exact Test](../cd/methods/fetdrift.ipynb)                            |        ✅         |        ✅         |
| [Least-Squares Density Difference](../cd/methods/lsdddrift.ipynb)              |        ❌         |        ✅         |
| [Maximum Mean Discrepancy](../cd/methods/mmddrift.ipynb)                       |        ✅         |        ✅         |
| [Learned Kernel MMD](../cd/methods/learnedkerneldrift.ipynb)                   |        ❌         |        ✅         |
| [Chi-Squared](../cd/methods/chisquaredrift.ipynb)                              |        ✅         |        ✅         |
| [Mixed-type tabular](../cd/methods/tabulardrift.ipynb)                         |        ✅         |        ✅         |
| [Classifier](../cd/methods/classifierdrift.ipynb)                              |        ✅         |        ✅         |
| [Spot-the-diff](../cd/methods/spotthediffdrift.ipynb)                          |        ❌         |        ✅         |
| [Classifier Uncertainty](../cd/methods/modeluncdrift.ipynb)                    |        ❌         |        ✅         |
| [Regressor Uncertainty](../cd/methods/modeluncdrift.ipynb)                     |        ❌         |        ✅         |
| [Online Cramér-von Mises](../cd/methods/onlinecvmdrift.ipynb)                  |        ❌         |        ✅         |
| [Online Fisher's Exact Test](../cd/methods/onlinefetdrift.ipynb)               |        ❌         |        ✅         |
| [Online Least-Squares Density Difference](../cd/methods/onlinelsdddrift.ipynb) |        ❌         |        ✅         |
| [Online Maximum Mean Discrepancy](../cd/methods/onlinemmddrift.ipynb)          |        ❌         |        ✅         |
```

```{tab-item} Outlier detectors
| Detector                                                | Legacy save/load | Config save/load |
|:--------------------------------------------------------|:----------------:|:----------------:|
| [Isolation Forest](../od/methods/iforest.ipynb)         |         ✅       |       ❌          |         
| [Mahalanobis Distance](../od/methods/mahalanobis.ipynb) |         ✅       |       ❌          |
| [AE](../od/methods/ae.ipynb)                            |         ✅       |       ❌          |
| [VAE](../od/methods/vae.ipynb)                          |         ✅       |       ❌          |
| [AEGMM](../od/methods/aegmm.ipynb)                      |         ✅       |       ❌          |
| [VAEGMM](../od/methods/vaegmm.ipynb)                    |         ✅       |       ❌          |
| [Likelihood Ratios](../od/methods/llr.ipynb)            |         ✅       |       ❌          |
| [Prophet](../od/methods/prophet.ipynb)                  |         ✅       |       ❌          |
| [Spectral Residual](../od/methods/sr.ipynb)             |         ✅       |       ❌          |
| [Seq2Seq](../od/methods/seq2seq.ipynb)                  |         ✅       |       ❌          |

```

```{tab-item} Adversarial detectors
| Detector                                                    | Legacy save/load | Config save/load |
|:------------------------------------------------------------|:----------------:|:----------------:|
| [Adversarial AE](../ad/methods/adversarialae.ipynb)         |        ✅        |        ❌         |
| [Model distillation](../ad/methods/modeldistillation.ipynb) |        ✅        |        ❌         |
```
````

```{note}
Saving/loading of detectors using PyTorch models and/or a PyTorch backend is currently not supported. 
```
