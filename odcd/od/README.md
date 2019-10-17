## Outlier Detection

Proposed format:

```python
class OutlierDetector:
  def __init__(self, threshold, *args, **kwargs)
    # Initialize outlier detector.
    self.threshold = threshold
  
  def fit(self, X, *args, **kwargs):
    # Optional
  
  def score(self, X, *args, **kwargs):
    # Compute outlier scores of X.
    return outlier_score
   
  def predict(self, X, *args, **kwargs):
    # Compute outlier scores and transform into outlier predictions.
    outlier_score = self.score(X, *args, **kwargs)
    outlier_pred = (outlier_score > self.threshold).astype(int)
    return outlier_pred
```
