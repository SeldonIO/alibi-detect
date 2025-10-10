# `alibi_detect.od.pytorch.svm`
## `BgdSVMTorch`

_Inherits from:_ `SVMTorch`, `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
BgdSVMTorch(self, nu: float, kernel: 'torch.nn.Module' = None, n_components: Optional[int] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nu` | `float` |  | The proportion of the training data that should be considered outliers. Note that this does not necessarily correspond to the false positive rate on test data, which is still defined when calling the `infer_threshold` method. |
| `kernel` | `Optional[torch.nn.modules.module.Module]` | `None` | Kernel function to use for outlier detection. |
| `n_components` | `Optional[int]` | `None` | Number of components in the Nystroem approximation, by default uses all of them. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |

### Methods

#### `fit`

```python
fit(x_ref: torch.Tensor, step_size_range: Tuple[float, float] = (1e-08, 1.0), n_step_sizes: int = 16, tol: float = 1e-06, n_iter_no_change: int = 25, max_iter: int = 1000, verbose: int = 0) -> Dict
```

Fit the Nystroem approximation and python SVM model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `torch.Tensor` |  | Training data. |
| `step_size_range` | `Tuple[float, float]` | `(1e-08, 1.0)` | The range of values to be considered for the gradient descent step size at each iteration. This is specified as a tuple of the form `(min_eta, max_eta)`. |
| `n_step_sizes` | `int` | `16` | The number of step sizes in the defined range to be tested for loss reduction. This many points are spaced equidistantly along the range in log space. |
| `tol` | `float` | `1e-06` | The decrease in loss required over the previous n_iter_no_change iterations in order to continue optimizing. |
| `n_iter_no_change` | `int` | `25` | The number of iterations over which the loss must decrease by `tol` in order for optimization to continue. |
| `max_iter` | `int` | `1000` | The maximum number of optimization steps. |
| `verbose` | `int` | `0` | Verbosity level during training. ``0`` is silent, ``1`` a progress bar. |

**Returns**
- Type: `Dict`

#### `format_fit_kwargs`

```python
format_fit_kwargs(fit_kwargs: Dict) -> Dict
```

Format kwargs for `fit` method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fit_kwargs` | `Dict` |  | dictionary of Kwargs to format. See `fit` method for details. |

**Returns**
- Type: `Dict`

#### `score`

```python
score(x: torch.Tensor) -> torch.Tensor
```

Computes the score of `x`

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | `torch.Tensor` with leading batch dimension. |

**Returns**
- Type: `torch.Tensor`

## `SVMTorch`

_Inherits from:_ `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
SVMTorch(self, nu: float, kernel: 'torch.nn.Module' = None, n_components: Optional[int] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nu` | `float` |  | The proportion of the training data that should be considered outliers. Note that this does not necessarily correspond to the false positive rate on test data, which is still defined when calling the `infer_threshold` method. |
| `kernel` | `Optional[torch.nn.modules.module.Module]` | `None` | Kernel function to use for outlier detection. |
| `n_components` | `Optional[int]` | `None` | Number of components in the Nystroem approximation, by default uses all of them. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Detect if `x` is an outlier.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | `torch.Tensor` with leading batch dimension. |

**Returns**
- Type: `torch.Tensor`

## `SgdSVMTorch`

_Inherits from:_ `SVMTorch`, `TorchOutlierDetector`, `Module`, `FitMixinTorch`, `ABC`

### Constructor

```python
SgdSVMTorch(self, nu: float, kernel: 'torch.nn.Module' = None, n_components: Optional[int] = None, device: Union[typing_extensions.Literal['cuda', 'gpu', 'cpu'], ForwardRef('torch.device'), NoneType] = None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nu` | `float` |  | The proportion of the training data that should be considered outliers. Note that this does not necessarily correspond to the false positive rate on test data, which is still defined when calling the `infer_threshold` method. |
| `kernel` | `Optional[torch.nn.modules.module.Module]` | `None` | Kernel function to use for outlier detection. |
| `n_components` | `Optional[int]` | `None` | Number of components in the Nystroem approximation, by default uses all of them. |
| `device` | `Union[Literal[cuda, gpu, cpu], torch.device, None]` | `None` | Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``. |

### Methods

#### `fit`

```python
fit(x_ref: torch.Tensor, tol: float = 1e-06, max_iter: int = 1000, verbose: int = 0) -> Dict
```

Fit the Nystroem approximation and Sklearn `SGDOneClassSVM` SVM model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `torch.Tensor` |  | Training data. |
| `tol` | `float` | `1e-06` | The decrease in loss required over the previous ``n_iter_no_change`` iterations in order to continue optimizing. |
| `max_iter` | `int` | `1000` | The maximum number of optimization steps. |
| `verbose` | `int` | `0` | Verbosity level during training. ``0`` is silent, ``1`` a progress bar. |

**Returns**
- Type: `Dict`

#### `format_fit_kwargs`

```python
format_fit_kwargs(fit_kwargs: Dict) -> Dict
```

Format kwargs for `fit` method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fit_kwargs` | `Dict` |  | dictionary of Kwargs to format. See `fit` method for details. |

**Returns**
- Type: `Dict`

#### `score`

```python
score(x: torch.Tensor) -> torch.Tensor
```

Computes the score of `x`

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | `torch.Tensor` with leading batch dimension. |

**Returns**
- Type: `torch.Tensor`
