# `alibi_detect.od.sklearn.gmm`
## `GMMSklearn`

_Inherits from:_ `SklearnOutlierDetector`, `FitMixinSklearn`, `ABC`

### Constructor

```python
GMMSklearn(self, n_components: int)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_components` | `int` |  | Number of components in gaussian mixture model. |

### Methods

#### `fit`

```python
fit(x_ref: numpy.ndarray, tol: float = 0.001, max_iter: int = 100, n_init: int = 1, init_params: str = 'kmeans', verbose: int = 0) -> Dict
```

Fit the SKLearn GMM model`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x_ref` | `numpy.ndarray` |  | Reference data. |
| `tol` | `float` | `0.001` | Convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold. |
| `max_iter` | `int` | `100` | Maximum number of EM iterations to perform. |
| `n_init` | `int` | `1` | Number of initializations to perform. |
| `init_params` | `str` | `'kmeans'` | Method used to initialize the weights, the means and the precisions. Must be one of: 'kmeans' : responsibilities are initialized using kmeans. 'kmeans++' : responsibilities are initialized using kmeans++. 'random' : responsibilities are initialized randomly. 'random_from_data' : responsibilities are initialized randomly from the data. |
| `verbose` | `int` | `0` | Enable verbose output. If 1 then it prints the current initialization and each iteration step. If greater than 1 then it prints also the log probability and the time needed for each step. |

**Returns**
- Type: `Dict`

#### `format_fit_kwargs`

```python
format_fit_kwargs(fit_kwargs: Dict) -> Dict
```

Format kwargs for `fit` method.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fit_kwargs` | `Dict` |  |  |
| `kwargs` |  |  | dictionary of Kwargs to format. See `fit` method for details. |

**Returns**
- Type: `Dict`

#### `score`

```python
score(x: numpy.ndarray) -> numpy.ndarray
```

Computes the score of `x`

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | `np.ndarray` with leading batch dimension. |

**Returns**
- Type: `numpy.ndarray`
