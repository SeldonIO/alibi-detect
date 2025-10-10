# `alibi_detect.cd.preprocess`
## Functions
### `classifier_uncertainty`

```python
classifier_uncertainty(x: Union[numpy.ndarray, list], model_fn: Callable, preds_type: str = 'probs', uncertainty_type: str = 'entropy', margin_width: float = 0.1) -> numpy.ndarray
```

Evaluate model_fn on x and transform predictions to prediction uncertainties.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `model_fn` | `Callable` |  | Function that evaluates a classification model on x in a single call (contains batching logic if necessary). |
| `preds_type` | `str` | `'probs'` | Type of prediction output by the model. Options are 'probs' (in [0,1]) or 'logits' (in [-inf,inf]). |
| `uncertainty_type` | `str` | `'entropy'` | Method for determining the model's uncertainty for a given instance. Options are 'entropy' or 'margin'. |
| `margin_width` | `float` | `0.1` | Width of the margin if uncertainty_type = 'margin'. The model is considered uncertain on an instance if the highest two class probabilities it assigns to the instance differ by less than margin_width. |

**Returns**
- Type: `numpy.ndarray`

### `regressor_uncertainty`

```python
regressor_uncertainty(x: Union[numpy.ndarray, list], model_fn: Callable, uncertainty_type: str = 'mc_dropout', n_evals: int = 25) -> numpy.ndarray
```

Evaluate model_fn on x and transform predictions to prediction uncertainties.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `Union[numpy.ndarray, list]` |  | Batch of instances. |
| `model_fn` | `Callable` |  | Function that evaluates a regression model on x in a single call (contains batching logic if necessary). |
| `uncertainty_type` | `str` | `'mc_dropout'` | Method for determining the model's uncertainty for a given instance. Options are 'mc_dropout' or 'ensemble'. The former should output a scalar per instance. The latter should output a vector of predictions per instance. |
| `n_evals` | `int` | `25` | The number of times to evaluate the model under different dropout configurations. Only relavent when using the 'mc_dropout' uncertainty type. |

**Returns**
- Type: `numpy.ndarray`
