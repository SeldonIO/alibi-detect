# `alibi_detect.datasets`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.datasets (WARNING)>
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
### `corruption_types_cifar10c`

```python
corruption_types_cifar10c() -> List[str]
```

Retrieve list with corruption types used in CIFAR-10-C.

**Returns**
- Type: `List[str]`

### `fetch_attack`

```python
fetch_attack(dataset: str, model: str, attack: str, return_X_y: bool = False) -> Union[alibi_detect.utils.data.Bunch, Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]]
```

Load adversarial instances for a given dataset, model and attack type.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `dataset` | `str` |  | Dataset under attack. |
| `model` | `str` |  | Model under attack. |
| `attack` | `str` |  | Attack name. |
| `return_X_y` | `bool` | `False` | Bool, whether to only return the data and target values or a Bunch object. |

**Returns**
- Type: `Union[alibi_detect.utils.data.Bunch, Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]]`

### `fetch_cifar10c`

```python
fetch_cifar10c(corruption: Union[str, List[str]], severity: int, return_X_y: bool = False) -> Union[alibi_detect.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

Fetch CIFAR-10-C data. Originally obtained from https://zenodo.org/record/2535967#.XkKh2XX7Qts and

introduced in "Hendrycks, D and Dietterich, T.G. Benchmarking Neural Network Robustness to Common Corruptions
and Perturbations. In 7th International Conference on Learning Represenations, 2019.".

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `corruption` | `Union[str, List[str]]` |  | Corruption type. Options can be checked with `get_corruption_cifar10c()`. Alternatively, specify 'all' for all corruptions at a severity level. |
| `severity` | `int` |  | Severity level of corruption (1-5). |
| `return_X_y` | `bool` | `False` | Bool, whether to only return the data and target values or a Bunch object. |

**Returns**
- Type: `Union[alibi_detect.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`

### `fetch_ecg`

```python
fetch_ecg(return_X_y: bool = False) -> Union[alibi_detect.utils.data.Bunch, Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]]
```

Fetch ECG5000 data. The dataset contains 5000 ECG's, originally obtained from

Physionet (https://archive.physionet.org/cgi-bin/atm/ATM) under the name
"BIDMC Congestive Heart Failure Database(chfdb)", record "chf07".

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_X_y` | `bool` | `False` | Bool, whether to only return the data and target values or a Bunch object. |

**Returns**
- Type: `Union[alibi_detect.utils.data.Bunch, Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]]`

### `fetch_genome`

```python
fetch_genome(return_X_y: bool = False, return_labels: bool = False) -> Union[alibi_detect.utils.data.Bunch, tuple]
```

Load genome data including their labels and whether they are outliers or not. More details about the data can be

found in the readme on https://console.cloud.google.com/storage/browser/seldon-datasets/genome/.
The original data can be found here: https://drive.google.com/drive/folders/1Ht9xmzyYPbDouUTl_KQdLTJQYX2CuclR.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_X_y` | `bool` | `False` | Bool, whether to only return the data and target values or a Bunch object. |
| `return_labels` | `bool` | `False` | Whether to return the genome labels which are detailed in the `label_json` dict of the returned Bunch object. |

**Returns**
- Type: `Union[alibi_detect.utils.data.Bunch, tuple]`

### `fetch_kdd`

```python
fetch_kdd(target: list = ['dos', 'r2l', 'u2r', 'probe'], keep_cols: list = ['srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'], percent10: bool = True, return_X_y: bool = False) -> Union[alibi_detect.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

KDD Cup '99 dataset. Detect computer network intrusions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `target` | `list` | `['dos', 'r2l', 'u2r', 'probe']` | List with attack types to detect. |
| `keep_cols` | `list` | `['srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']` | List with columns to keep. Defaults to continuous features. |
| `percent10` | `bool` | `True` | Bool, whether to only return 10% of the data. |
| `return_X_y` | `bool` | `False` | Bool, whether to only return the data and target values or a Bunch object. |

**Returns**
- Type: `Union[alibi_detect.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`

### `fetch_nab`

```python
fetch_nab(ts: str, return_X_y: bool = False) -> Union[alibi_detect.utils.data.Bunch, Tuple[pandas.core.frame.DataFrame, pandas.core.frame.DataFrame]]
```

Get time series in a DataFrame from the Numenta Anomaly Benchmark: https://github.com/numenta/NAB.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `ts` | `str` |  |  |
| `return_X_y` | `bool` | `False` | Bool, whether to only return the data and target values or a Bunch object. |

**Returns**
- Type: `Union[alibi_detect.utils.data.Bunch, Tuple[pandas.core.frame.DataFrame, pandas.core.frame.DataFrame]]`

### `get_list_nab`

```python
get_list_nab() -> list
```

Get list of possible time series to retrieve from the Numenta Anomaly Benchmark: https://github.com/numenta/NAB.

**Returns**
- Type: `list`

### `google_bucket_list`

```python
google_bucket_list(url: str, folder: str, filetype: Optional[str] = None, full_path: bool = False) -> List[str]
```

Retrieve list with items in google bucket folder.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | Bucket directory. |
| `folder` | `str` |  | Folder to retrieve list of items from. |
| `filetype` | `Optional[str]` | `None` | File extension, e.g. `npy` for saved numpy arrays. |
| `full_path` | `bool` | `False` |  |

**Returns**
- Type: `List[str]`

### `load_genome_npz`

```python
load_genome_npz(fold: str, return_labels: bool = False) -> Union[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fold` | `str` |  |  |
| `return_labels` | `bool` | `False` |  |

**Returns**
- Type: `Union[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]`

### `load_url_arff`

```python
load_url_arff(url: str, dtype: type[numpy.generic] = <class 'numpy.float32'>) -> numpy.ndarray
```

Load arff files from url.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url` | `str` |  | Address of arff file. |
| `dtype` | `type[numpy.generic]` | `<class 'numpy.float32'>` |  |

**Returns**
- Type: `numpy.ndarray`
