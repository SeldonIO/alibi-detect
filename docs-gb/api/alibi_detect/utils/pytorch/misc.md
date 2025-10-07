# `alibi_detect.utils.pytorch.misc`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi_detect.utils.pytorch.misc (WARNING)>
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

## Functions
### `get_device`

```python
get_device(device: Union[Literal[cuda, gpu, cpu], torch.device, None] = None) -> torch.device
```

Instantiates a PyTorch device object.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Either `None`, a str ('gpu', 'cuda' or 'cpu') indicating the device to choose, or an already instantiated device object. If `None`, the GPU is selected if it is detected, otherwise the CPU is used as a fallback. |

**Returns**
- Type: `torch.device`

### `get_optimizer`

```python
get_optimizer(name: str = 'Adam') -> type[torch.optim.optimizer.Optimizer]
```

Get an optimizer class from its name.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `name` | `str` | `'Adam'` | Name of the optimizer. |

**Returns**
- Type: `type[torch.optim.optimizer.Optimizer]`

### `quantile`

```python
quantile(sample: torch.Tensor, p: float, type: int = 7, sorted: bool = False) -> float
```

Estimate a desired quantile of a univariate distribution from a vector of samples

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `sample` | `torch.Tensor` |  | A 1D vector of values |
| `p` | `float` |  | The desired quantile in (0,1) |
| `type` | `int` | `7` | The method for computing the quantile. See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample |
| `sorted` | `bool` | `False` | Whether or not the vector is already sorted into ascending order |

**Returns**
- Type: `float`

### `zero_diag`

```python
zero_diag(mat: torch.Tensor) -> torch.Tensor
```

Set the diagonal of a matrix to 0

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `mat` | `torch.Tensor` |  | A 2D square matrix |

**Returns**
- Type: `torch.Tensor`
