# `alibi_detect.utils.fetching.fetching`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.utils.fetching.fetching (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

### `TIMEOUT`
```python
TIMEOUT: int = 10
```
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

## Functions
### `fetch_ad_ae`

```python
fetch_ad_ae(url: str, filepath: Union[str, os.PathLike], state_dict: dict) -> None
```

Download AE adversarial detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |
| `state_dict` | `dict` |  | Dictionary containing the detector's parameters. |

**Returns**
- Type: `None`

### `fetch_ad_md`

```python
fetch_ad_md(url: str, filepath: Union[str, os.PathLike]) -> None
```

Download model and distilled model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |

**Returns**
- Type: `None`

### `fetch_ae`

```python
fetch_ae(url: str, filepath: Union[str, os.PathLike]) -> None
```

Download AE outlier detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |

**Returns**
- Type: `None`

### `fetch_aegmm`

```python
fetch_aegmm(url: str, filepath: Union[str, os.PathLike]) -> None
```

Download AEGMM outlier detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |

**Returns**
- Type: `None`

### `fetch_detector`

```python
fetch_detector(filepath: Union[str, os.PathLike], detector_type: str, dataset: str, detector_name: str, model: str = None) -> Union[ForwardRef('BaseDetector'), ForwardRef('AdversarialAE'), ForwardRef('ModelDistillation'), ForwardRef('IForest'), ForwardRef('LLR'), ForwardRef('Mahalanobis'), ForwardRef('OutlierAEGMM'), ForwardRef('OutlierAE'), ForwardRef('OutlierProphet'), ForwardRef('OutlierSeq2Seq'), ForwardRef('OutlierVAE'), ForwardRef('OutlierVAEGMM'), ForwardRef('SpectralResidual')]
```

Fetch an outlier or adversarial detector from a google bucket, save it locally and return

the initialised detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |
| `detector_type` | `str` |  | `outlier` or `adversarial`. |
| `dataset` | `str` |  | Dataset of pre-trained detector. E.g. `kddcup`, `cifar10` or `ecg`. |
| `detector_name` | `str` |  | Name of the detector in the bucket. |
| `model` | `str` | `None` | Classification model used for adversarial detection. |

**Returns**
- Type: `Union[ForwardRef('BaseDetector'), ForwardRef('AdversarialAE'), ForwardRef('ModelDistillation'), ForwardRef('IForest'), ForwardRef('LLR'), ForwardRef('Mahalanobis'), ForwardRef('OutlierAEGMM'), ForwardRef('OutlierAE'), ForwardRef('OutlierProphet'), ForwardRef('OutlierSeq2Seq'), ForwardRef('OutlierVAE'), ForwardRef('OutlierVAEGMM'), ForwardRef('SpectralResidual')]`

### `fetch_enc_dec`

```python
fetch_enc_dec(url: str, filepath: Union[str, os.PathLike]) -> None
```

Download encoder and decoder networks.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |

**Returns**
- Type: `None`

### `fetch_llr`

```python
fetch_llr(url: str, filepath: Union[str, os.PathLike]) -> str
```

Download Likelihood Ratio outlier detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |

**Returns**
- Type: `str`

### `fetch_seq2seq`

```python
fetch_seq2seq(url: str, filepath: Union[str, os.PathLike]) -> None
```

Download sequence-to-sequence outlier detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |

**Returns**
- Type: `None`

### `fetch_state_dict`

```python
fetch_state_dict(url: str, filepath: Union[str, os.PathLike], save_state_dict: bool = True) -> Tuple[dict, dict]
```

Fetch the metadata and state/hyperparameter values of pre-trained detectors.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |
| `save_state_dict` | `bool` | `True` | Whether to save the state dict locally. |

**Returns**
- Type: `Tuple[dict, dict]`

### `fetch_tf_model`

```python
fetch_tf_model(dataset: str, model: str) -> keras.src.models.model.Model
```

Fetch pretrained tensorflow models from the google cloud bucket.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `dataset` | `str` |  | Dataset trained on. |
| `model` | `str` |  | Model name. |

**Returns**
- Type: `keras.src.models.model.Model`

### `fetch_vae`

```python
fetch_vae(url: str, filepath: Union[str, os.PathLike]) -> None
```

Download VAE outlier detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |

**Returns**
- Type: `None`

### `fetch_vaegmm`

```python
fetch_vaegmm(url: str, filepath: Union[str, os.PathLike]) -> None
```

Download VAEGMM outlier detector.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | URL to fetch detector from. |
| `filepath` | `Union[str, os.PathLike]` |  | Local directory to save detector to. |

**Returns**
- Type: `None`

### `get_pixelcnn_default_kwargs`

```python
get_pixelcnn_default_kwargs()
```
