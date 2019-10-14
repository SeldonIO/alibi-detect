## Data metrics
The purpose of the `DataTracker` component is to compute and store online statistics of the incoming
data distribution. It is intended for use in tabular datasets, but may be extended to other types
of data.

The initial intention is to use `creme` for the metric calculation and storage, but this could be
abstracted out and in the future other backends might be considered.

[Example notebook](../../examples/DataTracker.ipynb)

Proposed format:

```python
class DataTracker:

  def __init__(self, n_features, cat_vars)
  # initialize tracker based on the number of features and categorical features
  
  def update(self, X: np.ndarray):
  # receive a batch of data and update running metrics
  
  def get(self, serialize=True):
  # get the current values of the metrics