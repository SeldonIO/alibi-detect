---
title: Learned drift detectors on Adult Census
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---


Under the hood, drift detectors leverage a function (also known as a test-statistic) that is expected to take a large value if drift has occurred and a low value if not. The power of the detector is partly determined by how well the function satisfies this property. However, specifying such a function in advance can be very difficult. 

## Detecting drift with a learned classifier

The classifier-based drift detector simply tries to correctly distinguish instances from the reference data vs. the test set. The classifier is trained to output the probability that a given instance belongs to the test set. If the probabilities it assigns to unseen tests instances are significantly higher (as determined by a Kolmogorov-Smirnov test) than those it assigns to unseen reference instances then the test set must differ from the reference set and drift is flagged. To leverage all the available reference and test data, stratified cross-validation can be applied and the out-of-fold predictions are used for the significance test. Note that a new classifier is trained for each test set or even each fold within the test set.

### Backend

The method works with both the **PyTorch**, **TensorFlow**, and **Sklearn** frameworks. We will focus exclusively on the **Sklearn** backend in this notebook.

### Dataset
**Adult** dataset consists of 32,561 distributed over 2 classes based on whether the annual income is >50K. We evaluate drift on particular subsets of the data which are constructed based on the education level. As we will further discuss, our reference dataset will consist of people having a **low education** level, while our test dataset will consist of people having a **high education** level.

**Note**: we need to install ``alibi`` to fetch the ``adult`` dataset.


```{python}
!pip install alibi
```

```{python}
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from alibi.datasets import fetch_adult
from alibi_detect.cd import ClassifierDrift
```

### Load Adult Census Dataset

```{python}
# fetch adult dataset
adult = fetch_adult()

# separate columns in numerical and categorical.
categorical_names = [adult.feature_names[i] for i in adult.category_map.keys()]
categorical_ids = list(adult.category_map.keys())

numerical_names = [name for i, name in enumerate(adult.feature_names) if i not in adult.category_map.keys()]
numerical_ids = [i for i in range(len(adult.feature_names)) if i not in adult.category_map.keys()]

X = adult.data
```

We split the dataset in two based on the education level. We define a `low_education` level consisting of: `'Dropout'`, `'High School grad'`, `'Bachelors'`, and a `high_education` level consisting of: `'Bachelors'`, `'Masters'`, `'Doctorate'`. Intentionally we included an overlap between the two distributions consisting of people that have a `Bachelors` degree. Our goal is to detect that the two distributions are different.

```{python}
education_col = adult.feature_names.index('Education')
education = adult.category_map[education_col]
print(education)
```

```{python}
# define low education
low_education = [
    education.index('Dropout'),
    education.index('High School grad'),
    education.index('Bachelors')
    
]
# define high education
high_education = [
    education.index('Bachelors'),
    education.index('Masters'),
    education.index('Doctorate')
]
print("Low education:", [education[i] for i in low_education])
print("High education:", [education[i] for i in high_education])
```

```{python}
# select instances for low and high education
low_education_mask = pd.Series(X[:, education_col]).isin(low_education).to_numpy()
high_education_mask = pd.Series(X[:, education_col]).isin(high_education).to_numpy()
X_low, X_high = X[low_education_mask], X[high_education_mask]
```

We sample our reference dataset from the `low_education` level. In addition, we sample two other datasets:

 * `x_h0` - sampled from the `low_education` level to support the null hypothesis (i.e., the two distributions are identical);

* `x_h1` - sampled from the `high_education` level to support the alternative hypothesis (i.e., the two distributions are different);

```{python}
size = 1000
np.random.seed(0)

# define reference and H0 dataset
idx_low = np.random.choice(np.arange(X_low.shape[0]), size=2*size, replace=False)
x_ref, x_h0 = train_test_split(X_low[idx_low], test_size=0.5, random_state=5, shuffle=True)

# define reference and H1 dataset
idx_high = np.random.choice(np.arange(X_high.shape[0]), size=size, replace=False)
x_h1 = X_high[idx_high]
```

### Define dataset pre-processor

```{python}
# define numerical standard scaler.
num_transf = StandardScaler()

# define categorical one-hot encoder.
cat_transf = OneHotEncoder(
    categories=[range(len(x)) for x in adult.category_map.values()],
    handle_unknown="ignore"
)

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transf, categorical_ids),
        ("num", num_transf, numerical_ids),
    ],
    sparse_threshold=0
)

# fit preprocessor.
preprocessor = preprocessor.fit(np.concatenate([x_ref, x_h0, x_h1]))
```

### Utils

```{python}
labels = ['No!', 'Yes!']

def print_preds(preds: dict, preds_name: str) -> None:
    print(preds_name)
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print(f'p-value: {preds["data"]["p_val"]:.3f}')
    print('')
```

### Drift detection 

We perform a **binomial** test using a `RandomForestClassifier`. 

```{python}
# define classifier
model = RandomForestClassifier()

# define drift detector with binarize prediction
detector = ClassifierDrift(
    x_ref=x_ref,
    model=model,
    backend='sklearn',
    preprocess_fn=preprocessor.transform,
    binarize_preds=True,
    n_folds=2,
)

# print results
print_preds(detector.predict(x=x_h0), "H0")
print_preds(detector.predict(x=x_h1), "H1")
```

As expected, when testing against `x_h0`, we fail to reject $H_0$, while for the second case there is enough evidence to reject $H_0$ and flag that the data has drifted.

For the classifiers that do not support `predict_proba` but offer support for `decision_function`, we can perform a **K-S** test on the scores by setting `preds_type='scores'`.

```{python}
# define model
model = GradientBoostingClassifier()


# define drift detector
detector = ClassifierDrift(
    x_ref=x_ref,
    model=model,
    backend='sklearn',
    preprocess_fn=preprocessor.transform,
    preds_type='scores',
    binarize_preds=False,
    n_folds=2,
)

# print results
print_preds(detector.predict(x=x_h0), "H0")
print_preds(detector.predict(x=x_h1), "H1")
```

Some models can return a poor estimate of the class label probability or some might not even support probability predictions. We can add calibration on top of each classifier to obtain better probability estimates and perform a **K-S** test. For demonstrative purposes, we will calibrate a ``LinearSVC`` which does not support ``predict_proba``, but any other classifier would work.

```{python}
# define model - does not support predict_proba
model = LinearSVC(max_iter=10000)

# define drift detector
detector = ClassifierDrift(
    x_ref=x_ref,
    model=model,
    backend='sklearn',
    preprocess_fn=preprocessor.transform,
    binarize_preds=False,
    n_folds=2,
    use_calibration=True,
    calibration_kwargs={'method': 'isotonic'}
)

# print results
print_preds(detector.predict(x=x_h0), "H0")
print_preds(detector.predict(x=x_h1), "H1")
```

### Speeding things up

In order to use the entire dataset and obtain unbiased predictions required to perform the statistical test, the  `ClassifierDrift` detector has the option to perform a `n_folds` split. Although appealing due to its data efficiency, this method can be slow since it is required to train a number of `n_folds` classifiers. 

For the `RandomForestClassifier` we can avoid retraining `n_folds` classifiers by using the out-of-bag predictions. In a `RandomForestClassifier` each tree is trained on a separate dataset obtained by sampling with replacement the original training set, a method known as bagging. On average, only 63\% unique samples from the original dataset are used to train each tree ([Bostrom](https://people.dsv.su.se/~henke/papers/bostrom08b.pdf)). Thus, for each tree, we can obtain predictions for the remaining out-of-bag samples (i.e., the rest of 37\%). By cumulating the out-of-bag predictions across all the trees we can eventually obtain a prediction for each sample in the original dataset. Note that we used the word 'eventually' because if the number of trees is too small, covering the entire original dataset might be unlikely.

For demonstrative purposes, we will compare the running time of the `ClassifierDrift` detector when using a `RandomForestClassifier` in two setups: `n_folds=5, use_oob=False` and `use_oob=True`.

```{python}
n_estimators = 400
n_folds = 5
```

```{python}
%%time
# define drift detector
detector_rf = ClassifierDrift(
    x_ref=x_ref,
    model=RandomForestClassifier(n_estimators=n_estimators),
    backend='sklearn',
    preprocess_fn=preprocessor.transform,
    binarize_preds=False,
    n_folds=n_folds
)

# print results
print_preds(detector_rf.predict(x=x_h0), "H0")
print_preds(detector_rf.predict(x=x_h1), "H1")
```

```{python}
%%time
# define drift detector
detector_rf_oob = ClassifierDrift(
    x_ref=x_ref,
    model=RandomForestClassifier(n_estimators=n_estimators),
    backend='sklearn',
    preprocess_fn=preprocessor.transform,
    binarize_preds=False,
    use_oob=True
)

# print results
print_preds(detector_rf_oob.predict(x=x_h0), "H0")
print_preds(detector_rf_oob.predict(x=x_h1), "H1")
```

We can observe that in this particular setting, using the out-of-bag prediction can speed up the procedure up to almost x4.

