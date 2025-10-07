# `alibi_detect.saving.schemas`

Pydantic models used by :func:`~alibi_detect.utils.validate.validate_config` to validate configuration dictionaries.
The `resolved` kwarg of :func:`~alibi_detect.utils.validate.validate_config` determines whether the *unresolved* or
*resolved* pydantic models are used:

- The *unresolved* models expect any artefacts specified within it to not yet have been resolved.
  The artefacts are still string references to local filepaths or registries (e.g. `x_ref = 'x_ref.npy'`).

- The *resolved* models expect all artefacts to be have been resolved into runtime objects. For example, `x_ref`
  should have been resolved into an `np.ndarray`.

.. note::
    For detector pydantic models, the fields match the corresponding detector's args/kwargs. Refer to the
    detector's api docs for a full description of each arg/kwarg.

## Constants
### `supported_models_all`
```python
supported_models_all: tuple = (<class 'keras.src.models.model.Model'>, <class 'torch.nn.modules.module.Modu...
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_models_tf`
```python
supported_models_tf: tuple = (<class 'keras.src.models.model.Model'>,)
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_models_sklearn`
```python
supported_models_sklearn: tuple = (<class 'sklearn.base.BaseEstimator'>,)
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_models_torch`
```python
supported_models_torch: tuple = (<class 'torch.nn.modules.module.Module'>,)
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_optimizers_tf`
```python
supported_optimizers_tf: tuple = (<class 'keras.src.optimizers.optimizer.Optimizer'>, <class 'keras.src.optimi...
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_optimizers_torch`
```python
supported_optimizers_torch: tuple = (<class 'type'>,)
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `supported_optimizers_all`
```python
supported_optimizers_all: tuple = (<class 'keras.src.optimizers.optimizer.Optimizer'>, <class 'keras.src.optimi...
```
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `DETECTOR_CONFIGS`
```python
DETECTOR_CONFIGS: dict = {'KSDrift': <class 'alibi_detect.saving.schemas.KSDriftConfig'>, 'ChiSquareDr...
```

### `DETECTOR_CONFIGS_RESOLVED`
```python
DETECTOR_CONFIGS_RESOLVED: dict = {'KSDrift': <class 'alibi_detect.saving.schemas.KSDriftConfigResolved'>, 'Chi...
```

## `CVMDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`CVMDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/cvmdrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.CVMDrift` documentation for a description of each field.

## `CVMDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`CVMDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/cvmdrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.CVMDrift` documentation for a description of each field.

## `CVMDriftOnlineConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`CVMDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinecvmdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.CVMDriftOnline` documentation for a description of each field.

## `CVMDriftOnlineConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`CVMDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinecvmdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.CVMDriftOnline` documentation for a description of each field.

## `ChiSquareDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`ChiSquareDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/chisquaredrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.ChiSquareDrift` documentation for a description of each field.

## `ChiSquareDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`ChiSquareDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/chisquaredrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.ChiSquareDrift` documentation for a description of each field.

## `ClassifierDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`ClassifierDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.ClassifierDrift` documentation for a description of each field.

## `ClassifierDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`ClassifierDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.ClassifierDrift` documentation for a description of each field.

## `ClassifierUncertaintyDriftConfig`

_Inherits from:_ `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`ClassifierUncertaintyDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/modeluncdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.ClassifierUncertaintyDrift` documentation for a description of each field.

## `ClassifierUncertaintyDriftConfigResolved`

_Inherits from:_ `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`ClassifierUncertaintyDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/modeluncdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.ClassifierUncertaintyDrift` documentation for a description of each field.

## `ContextMMDDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`ContextMMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/contextmmddrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.ContextMMDDrift` documentation for a description of each field.

## `ContextMMDDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`MMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.MMDDrift` documentation for a description of each field.

## `CustomBaseModel`

_Inherits from:_ `BaseModel`, `Representation`

Base pydantic model schema. The default pydantic settings are set here.

## `CustomBaseModelWithKwargs`

_Inherits from:_ `BaseModel`, `Representation`

Base pydantic model schema. The default pydantic settings are set here.

## `DeepKernelConfig`

_Inherits from:_ `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for :class:`~alibi_detect.utils.tensorflow.kernels.DeepKernel`'s.

## `DetectorConfig`

_Inherits from:_ `CustomBaseModel`, `BaseModel`, `Representation`

Base detector config schema. Only fields universal across all detectors are defined here.

## `DriftDetectorConfig`

_Inherits from:_ `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved base schema for drift detectors.

## `DriftDetectorConfigResolved`

_Inherits from:_ `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved base schema for drift detectors.

## `EmbeddingConfig`

_Inherits from:_ `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for text embedding models. Currently, only pre-trained

`HuggingFace transformer <https://github.com/huggingface/transformers>`_ models are supported.

## `FETDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`FETDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/fetdrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.FETDrift` documentation for a description of each field.

## `FETDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`FETDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/fetdrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.FETDrift` documentation for a description of each field.

## `FETDriftOnlineConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`FETDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinefetdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.FETDriftOnline` documentation for a description of each field.

## `FETDriftOnlineConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`FETDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinefetdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.FETDriftOnline` documentation for a description of each field.

## `KSDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`KSDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.KSDrift` documentation for a description of each field.

## `KSDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`KSDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.KSDrift` documentation for a description of each field.
Resolved schema for the :class:`~alibi_detect.cd.KSDrift` detector.

## `KernelConfig`

_Inherits from:_ `CustomBaseModelWithKwargs`, `BaseModel`, `Representation`

Unresolved schema for kernels, to be passed to a detector's `kernel` kwarg.

If `src` specifies a :class:`~alibi_detect.utils.tensorflow.GaussianRBF` kernel, the `sigma`, `trainable` and
`init_sigma_fn` fields are passed to it. Otherwise, all fields except `src` are passed as kwargs.

## `LSDDDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`LSDDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/lsdddrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.LSDDDrift` documentation for a description of each field.

## `LSDDDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`LSDDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/lsdddrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.LSDDDrift` documentation for a description of each field.

## `LSDDDriftOnlineConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`LSDDDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinelsdddrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.LSDDDriftOnline` documentation for a description of each field.

## `LSDDDriftOnlineConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`LSDDDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinelsdddrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.LSDDDriftOnline` documentation for a description of each field.

## `LearnedKernelDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`LearnedKernelDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/learnedkerneldrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.LearnedKernelDrift` documentation for a description of each field.

## `LearnedKernelDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`LearnedKernelDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/learnedkerneldrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.LearnedKernelDrift` documentation for a description of each field.

## `MMDDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`MMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.MMDDrift` documentation for a description of each field.

## `MMDDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`MMDDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.MMDDrift` documentation for a description of each field.

## `MMDDriftOnlineConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`MMDDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinemmddrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.MMDDriftOnline` documentation for a description of each field.

## `MMDDriftOnlineConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`MMDDriftOnline <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinemmddrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.MMDDriftOnline` documentation for a description of each field.

## `MetaData`

_Inherits from:_ `CustomBaseModel`, `BaseModel`, `Representation`

## `ModelConfig`

_Inherits from:_ `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for (ML) models. Note that the model "backend" e.g. 'tensorflow', 'pytorch', 'sklearn', is set

by `backend` in :class:`DetectorConfig`.

## `OptimizerConfig`

_Inherits from:_ `CustomBaseModelWithKwargs`, `BaseModel`, `Representation`

Unresolved schema for optimizers. The `optimizer` dictionary has two possible formats:

1. A configuration dictionary compatible with
`tf.keras.optimizers.deserialize <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/deserialize>`_.
For `backend='tensorflow'` only.
2. A dictionary containing only `class_name`, where this is a string referencing the optimizer name e.g.
`optimizer.class_name = 'Adam'`. In this case, the tensorflow or pytorch optimizer class of the same name is
loaded. For `backend='tensorflow'` and `backend='pytorch'`.

## `PreprocessConfig`

_Inherits from:_ `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for drift detector preprocess functions, to be passed to a detector's `preprocess_fn` kwarg.

Once loaded, the function is wrapped in a :func:`~functools.partial`, to be evaluated within the detector.

If `src` specifies a generic Python function, the dictionary specified by `kwargs` is passed to it. Otherwise,
if `src` specifies :func:`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift`
(`src='@cd.tensorflow.preprocess.preprocess_drift'`), all fields (except `kwargs`) are passed to it.

## `RegressorUncertaintyDriftConfig`

_Inherits from:_ `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`RegressorUncertaintyDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/modeluncdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.RegressorUncertaintyDrift` documentation for a description of each field.

## `RegressorUncertaintyDriftConfigResolved`

_Inherits from:_ `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`RegressorUncertaintyDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/modeluncdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.RegressorUncertaintyDrift` documentation for a description of each field.

## `SpotTheDiffDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`SpotTheDiffDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.SpotTheDiffDrift` documentation for a description of each field.

## `SpotTheDiffDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`SpotTheDiffDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/spotthediffdrift.html>`_
detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.SpotTheDiffDrift` documentation for a description of each field.

## `SupportedDevice`

Pydantic custom type to check the device is correct for the choice of backend (conditional on what optional deps

are installed).

### Constructor

```python
SupportedDevice(self, /, *args, **kwargs)
```
### Methods

#### `validate_device`

```python
validate_device(device: typing.Any, values: dict) -> typing.Any
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `device` | `typing.Any` |  |  |
| `values` | `dict` |  |  |

**Returns**
- Type: `typing.Any`

## `SupportedModel`

Pydantic custom type to check the model is one of the supported types (conditional on what optional deps

are installed).

### Constructor

```python
SupportedModel(self, /, *args, **kwargs)
```
### Methods

#### `validate_model`

```python
validate_model(model: typing.Any, values: dict) -> typing.Any
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `typing.Any` |  |  |
| `values` | `dict` |  |  |

**Returns**
- Type: `typing.Any`

## `SupportedOptimizer`

Pydantic custom type to check the optimizer is one of the supported types (conditional on what optional deps

are installed).

### Constructor

```python
SupportedOptimizer(self, /, *args, **kwargs)
```
### Methods

#### `validate_optimizer`

```python
validate_optimizer(optimizer: typing.Any, values: dict) -> typing.Any
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `optimizer` | `typing.Any` |  |  |
| `values` | `dict` |  |  |

**Returns**
- Type: `typing.Any`

## `TabularDriftConfig`

_Inherits from:_ `DriftDetectorConfig`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for the

`TabularDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/tabulardrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.TabularDrift` documentation for a description of each field.

## `TabularDriftConfigResolved`

_Inherits from:_ `DriftDetectorConfigResolved`, `DetectorConfig`, `CustomBaseModel`, `BaseModel`, `Representation`

Resolved schema for the

`TabularDrift <https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/tabulardrift.html>`_ detector.

Except for the `name` and `meta` fields, the fields match the detector's args and kwargs. Refer to the
:class:`~alibi_detect.cd.TabularDrift` documentation for a description of each field.

## `TokenizerConfig`

_Inherits from:_ `CustomBaseModel`, `BaseModel`, `Representation`

Unresolved schema for text tokenizers. Currently, only pre-trained

`HuggingFace tokenizer <https://github.com/huggingface/tokenizers>`_ models are supported.
