---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

[source](../../api/alibi_detect.cd.model_uncertainty.rst)

# Model Uncertainty

## Overview

Model-uncertainty drift detectors aim to directly detect drift that's likely to effect the performance of a model of interest. The approach is to test for change in the number of instances falling into regions of the input space on which the model is uncertain in its predictions. For each instance in the reference set the detector obtains the model's prediction and some associated notion of uncertainty. For example for a classifier this may be the entropy of the predicted label probabilities or for a regressor with dropout layers dropout Monte Carlo can be used to provide a notion of uncertainty. The same is done for the test set and if significant differences in uncertainty are detected (via a Kolmogorov-Smirnoff test) then drift is flagged. The detector's reference set should be disjoint from the model's training set (on which the model's confidence may be higher).

`ClassifierUncertaintyDrift` should be used with classification models whereas `RegressorUncertaintyDrift` should be used with regression models. They are used in much the same way. 

By default `ClassifierUncertaintyDrift` uses `uncertainty_type='entropy'` as the notion of uncertainty for classifier predictions and a [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) two-sample test is performed on these continuous values. However `uncertainty_type='margin'` can also be specified to deem the classifier's prediction uncertain if they fall within a margin (e.g. in \[0.45,0.55\] for binary classifier probabilities) (similar to [Sethi and Kantardzic (2017)](https://arxiv.org/abs/1704.00023)) and a [Chi-Squared](https://en.wikipedia.org/wiki/Chi-squared_test) two-sample test is performed on these 0-1 flags of uncertainty.

By default `RegressorUncertaintyDrift` uses `uncertainty_type='mc_dropout'` and assumes a PyTorch or TensorFlow model with dropout layers as the regressor. This evaluates the model under multiple dropout configurations and uses the variation as the notion of uncertainty. Alternatively a model that outputs (for each instance) a vector of independent model predictions can be passed and `uncertainty_type='ensemble'` can be specified. Again the variation is taken as the notion of uncertainty and in both cases a Kolmogorov-Smirnov two-sample test is performed on the continuous notions of uncertainty.

## Usage

### Initialize

Arguments:

* `x_ref`: Data used as reference distribution. Should be disjoint from the model's training set

* `model`: The model of interest whose performance we'd like to remain constant.

Keyword arguments:

* `p_val`: p-value used for the significance of the test.

* `update_x_ref`: Reference data can optionally be updated to the last N instances seen by the detector or via [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) with size N. For the former, the parameter equals *{'last': N}* while for reservoir sampling *{'reservoir_sampling': N}* is passed.

* `input_shape`: Optionally pass the shape of the input data.

* `data_type`: Optionally specify the data type (e.g. tabular, image or time-series). Added to metadata.

`ClassifierUncertaintyDrift`-specific keyword arguments:

* `preds_type`: Type of prediction output by the model. Options are 'probs' (in \[0,1\]) or 'logits' (in \[-inf,inf\]).

* `uncertainty_type`: Method for determining the model's uncertainty for a given instance. Options are 'entropy' or 'margin'.

* `margin_width`: Width of the margin if uncertainty_type = 'margin'. The model is considered uncertain on an instance if the highest two class probabilities it assigns to the instance differ by less than this.

`RegressorUncertaintyDrift`-specific keyword arguments:

* `uncertainty_type`: Method for determining the model's uncertainty for a given instance. Options are 'mc_dropout' or 'ensemble'. For the former the model should have dropout layers and output a scalar per instance. For the latter the model should output a vector of predictions per instance.

* `n_evals`: The number of times to evaluate the model under different dropout configurations. Only relavent when using the 'mc_dropout' uncertainty type.

Additional arguments if batch prediction required:

* `backend`: Framework that was used to define model. Options are 'tensorflow' or 'pytorch'.

* `batch_size`: Batch size to use to evaluate model. Defaults to 32.

* `device`: Device type to use. The default None tries to use the GPU and falls back on CPU if needed. Can be specified by passing either 'cuda', 'gpu' or 'cpu'. Only relevant for 'pytorch' backend.

Additional arguments for NLP models

* `tokenizer`: Tokenizer to use before passing data to model.

* `max_len`: Max length to be used by tokenizer.

### Examples

Drift detector for a **TensorFlow** classifier outputting probabilities:

```python
from alibi_detect.cd import ClassifierUncertaintyDrift

clf =  # tensorflow classifier model
cd = ClassifierUncertaintyDetector(x_ref, clf, backend='tensorflow', p_val=.05, preds_type='probs')
```

Drift detector for a **PyTorch** regressor (with dropout layers) outputting scalars:

```python
from alibi_detect.cd import RegressorUncertaintyDrift

reg =  # pytorch regression model with at least 1 dropout layer
cd = RegressorUncertaintyDrift(x_ref, reg, backend='pytorch', p_val=.05, uncertainty_type='mc_dropout')
```

Note that for the PyTorch RegressorUncertaintyDrift detector the dropout layers need to be defined within the `nn.Module` init to be able to set them to train mode when computing the uncertainty estimates, e.g.:

```python
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # define model
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # do forward pass which includes self.dropout
```

### Detect Drift

We detect data drift by simply calling `predict` on a batch of instances `x`. `return_p_val` equal to *True* will also return the p-value of the test and `return_distance` equal to *True* will return the test-statistic.

The prediction takes the form of a dictionary with `meta` and `data` keys. `meta` contains the detector's metadata while `data` is also a dictionary which contains the actual predictions stored in the following keys:

* `is_drift`: 1 if the sample tested has drifted from the reference data and 0 otherwise.

* `threshold`: the user-defined threshold defining the significance of the test.

* `p_val`: the p-value of the test if `return_p_val` equals *True*.

* `distance`: the test-statistic if `return_distance` equals *True*.


```python
preds = cd.predict(x)
```

## Examples

### Graph

[Drift detection on molecular graphs](../../examples/cd_mol.ipynb)

### Image and tabular

[Drift detection on CIFAR10 and Wine Quality Data Set](../../examples/cd_model_unc_cifar10_wine.ipynb)

