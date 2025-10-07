# `alibi_detect.utils.visualize`
## Functions
### `plot_feature_outlier_image`

```python
plot_feature_outlier_image(od_preds: Dict, X: numpy.ndarray, X_recon: Optional[numpy.ndarray] = None, instance_ids: Optional[list] = None, max_instances: int = 5, outliers_only: bool = False, n_channels: int = 3, figsize: tuple = (20, 20)) -> None
```

Plot feature (pixel) wise outlier scores for images.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `od_preds` | `Dict` |  | Output of an outlier detector's prediction. |
| `X` | `numpy.ndarray` |  | Batch of instances to apply outlier detection to. |
| `X_recon` | `Optional[numpy.ndarray]` | `None` | Reconstructed instances of X. |
| `instance_ids` | `Optional[list]` | `None` | List with indices of instances to display. |
| `max_instances` | `int` | `5` | Maximum number of instances to display. |
| `outliers_only` | `bool` | `False` | Whether to only show outliers or not. |
| `n_channels` | `int` | `3` | Number of channels of the images. |
| `figsize` | `tuple` | `(20, 20)` | Tuple for the figure size. |

**Returns**
- Type: `None`

### `plot_feature_outlier_tabular`

```python
plot_feature_outlier_tabular(od_preds: Dict, X: numpy.ndarray, X_recon: Optional[numpy.ndarray] = None, threshold: Optional[float] = None, instance_ids: Optional[list] = None, max_instances: int = 5, top_n: int = 1000000000000, outliers_only: bool = False, feature_names: Optional[list] = None, width: float = 0.2, figsize: tuple = (20, 10)) -> None
```

Plot feature wise outlier scores for tabular data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `od_preds` | `Dict` |  | Output of an outlier detector's prediction. |
| `X` | `numpy.ndarray` |  | Batch of instances to apply outlier detection to. |
| `X_recon` | `Optional[numpy.ndarray]` | `None` | Reconstructed instances of X. |
| `threshold` | `Optional[float]` | `None` | Threshold used for outlier score to determine outliers. |
| `instance_ids` | `Optional[list]` | `None` | List with indices of instances to display. |
| `max_instances` | `int` | `5` | Maximum number of instances to display. |
| `top_n` | `int` | `1000000000000` | Maixmum number of features to display, ordered by outlier score. |
| `outliers_only` | `bool` | `False` | Whether to only show outliers or not. |
| `feature_names` | `Optional[list]` | `None` | List with feature names. |
| `width` | `float` | `0.2` | Column width for bar charts. |
| `figsize` | `tuple` | `(20, 10)` | Tuple for the figure size. |

**Returns**
- Type: `None`

### `plot_feature_outlier_ts`

```python
plot_feature_outlier_ts(od_preds: Dict, X: numpy.ndarray, threshold: Union[float, int, list, numpy.ndarray], window: Optional[tuple] = None, t: Optional[numpy.ndarray] = None, X_orig: Optional[numpy.ndarray] = None, width: float = 0.2, figsize: tuple = (20, 8), ylim: tuple = (None, None)) -> None
```

Plot feature wise outlier scores for time series data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `od_preds` | `Dict` |  | Output of an outlier detector's prediction. |
| `X` | `numpy.ndarray` |  | Time series to apply outlier detection to. |
| `threshold` | `Union[float, int, list, numpy.ndarray]` |  | Threshold used to classify outliers or adversarial instances. |
| `window` | `Optional[tuple]` | `None` | Start and end timestep to plot. |
| `t` | `Optional[numpy.ndarray]` | `None` | Timesteps. |
| `X_orig` | `Optional[numpy.ndarray]` | `None` | Optional original time series without outliers. |
| `width` | `float` | `0.2` | Column width for bar charts. |
| `figsize` | `tuple` | `(20, 8)` | Tuple for the figure size. |
| `ylim` | `tuple` | `(None, None)` | Min and max y-axis values for the outlier scores. |

**Returns**
- Type: `None`

### `plot_instance_score`

```python
plot_instance_score(preds: Dict, target: numpy.ndarray, labels: numpy.ndarray, threshold: float, ylim: tuple = (None, None)) -> None
```

Scatter plot of a batch of outlier or adversarial scores compared to the threshold.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `preds` | `Dict` |  | Dictionary returned by predictions of an outlier or adversarial detector. |
| `target` | `numpy.ndarray` |  | Ground truth. |
| `labels` | `numpy.ndarray` |  | List with names of classification labels. |
| `threshold` | `float` |  | Threshold used to classify outliers or adversarial instances. |
| `ylim` | `tuple` | `(None, None)` | Min and max y-axis values. |

**Returns**
- Type: `None`

### `plot_roc`

```python
plot_roc(roc_data: Dict[str, Dict[str, numpy.ndarray]], figsize: tuple = (10, 5)) -> None
```

Plot ROC curve.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `roc_data` | `Dict[str, Dict[str, numpy.ndarray]]` |  | Dictionary with as key the label to show in the legend and as value another dictionary with as keys `scores` and `labels` with respectively the outlier scores and outlier labels. |
| `figsize` | `tuple` | `(10, 5)` | Figure size. |

**Returns**
- Type: `None`
