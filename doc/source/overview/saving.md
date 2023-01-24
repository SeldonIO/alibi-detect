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
| [Cramér-von Mises](../cd/methods/cvmdrift.ipynb)                               |        ❌         |        ✅         |
| [Fisher's Exact Test](../cd/methods/fetdrift.ipynb)                            |        ❌         |        ✅         |
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

(supported_models)=
## Supported ML models

Alibi Detect drift detectors offer the option to perform [preprocessing](../cd/background.md#input-preprocessing)
with user-defined machine learning models:

```python
model = ... # A TensorFlow model
preprocess_fn = partial(preprocess_drift, model=model, batch_size=128)
cd = MMDDrift(x_ref, backend='tensorflow', p_val=.05, preprocess_fn=preprocess_fn)
```

Additionally, some detectors are built upon models directly, 
for example the [Classifier](../cd/methods/classifierdrift.ipynb) drift detector requires a `model` to be passed
as an argument:

```python
cd = ClassifierDrift(x_ref, model, backend='sklearn', p_val=.05, preds_type='probs')
```

In order for a detector to be saveable and loadable, any models contained within it (or referenced within a 
[detector configuration file](config_files.md#specifying-artefacts)) must fall within the family of supported models:

````{tab-set}

```{tab-item} TensorFlow
Alibi Detect supports serialization of any TensorFlow model that can be serialized to the 
[HDF5](https://www.tensorflow.org/guide/keras/save_and_serialize#keras_h5_format) format. 
Custom objects should be pre-registered with 
[register_keras_serializable](https://www.tensorflow.org/api_docs/python/tf/keras/utils/register_keras_serializable).
```

```{tab-item} PyTorch
PyTorch models are serialized by saving the [entire model](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model)
using the [dill](https://dill.readthedocs.io/en/latest/index.html) module. Therefore, Alibi Detect should support any PyTorch 
model that can be saved and loaded with `torch.save(..., pickle_module=dill)` and `torch.load(..., pickle_module=dill)`.
```

```{tab-item} Scikit-learn
Scikit-learn models are serialized using [joblib](https://joblib.readthedocs.io/en/latest/persistence.html).
Any scikit-learn model that is a subclass of {py:class}`sklearn.base.BaseEstimator` is supported, including 
[xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) models following 
the scikit-learn API.
```
````

## Online detectors

[Online drift detectors](../cd/methods.md#online) are stateful, with their state updated each timestep `t` (each time
`.predict()` is called). {func}`~alibi_detect.saving.save_detector` will save the state of online 
detectors to disk if `t > 0`. At load time, {func}`~alibi_detect.saving.load_detector` will load this state.
For example:

```python
from alibi_detect.cd import LSDDDriftOnline
from alibi_detect.saving import save_detector, load_detector

# Init detector (t=0)
dd = LSDDDriftOnline(x_ref, window_size=10, ert=50)

# Run 2 predictions
pred_1 = dd.predict(x_1)  # t=1 
pred_2 = dd.predict(x_2)  # t=2

# Save detector (state will be saved since t>0)
save_detector(dd, filepath)

# Load detector
dd_new = load_detector(filepath)  # detector will start at t=2
```

To save a clean (stateless) detector, it should be reset before saving:

```python
dd.reset_state()  # reset to t=0
save_detector(dd, filepath)  # save the detector without state
```