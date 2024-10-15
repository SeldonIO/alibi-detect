---
title: Drift detection on Amazon reviews
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---



## Methods

We illustrate drift detection on text data using the following detectors:

- [Maximum Mean Discrepancy (MMD) detector](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html) using [pre-trained transformers](https://huggingface.co/transformers/) to flag drift in the embedding space.

- [Classifier drift detector](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html) to detect drift in the input space.


## Dataset

The *Amazon* dataset contains product reviews with a star rating. We will test whether drift can be detected if the ratings start to drift. For more information, check the [WILDS documentation page](https://wilds.stanford.edu/datasets/#amazon).


## Dependencies

Besides `alibi-detect`, this example notebook also uses the *Amazon* dataset through the [WILDS](https://wilds.stanford.edu/datasets/) package. WILDS is a curated collection of benchmark datasets that represent distribution shifts faced in the wild and can be installed via `pip`:


```{python}
!pip install wilds
```


Throughout the notebook we use detectors with both `PyTorch` and `TensorFlow` backends.

```{python}
import numpy as np
import torch

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

seed = 1234
set_seed(seed)
```

## Load and prepare data

We first load the dataset and create reference data, data which should not be rejected under the null of the test (H0) and data which should exhibit drift (H1). The drift is introduced later by specifying a specific star rating for the test instances.

```{python}
AMAZON_PATH = './data/amazon' # path to save data
DOWNLOAD = False  # set to True for first run
```

<div class="alert alert-warning">
The following cell will download the Amazon dataset (if DOWNLOAD=True). The download size is ~7GB and size on disk is ~7GB.
</div>

```{python}
#| scrolled: true
from functools import partial
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

ds = get_dataset(dataset='amazon', root_dir=AMAZON_PATH, download=DOWNLOAD)
ds_tr = ds.get_subset('train')
idx_ref, idx_h0 = train_test_split(np.arange(len(ds_tr)), train_size=.5, random_state=seed, shuffle=True)
ds_ref = Subset(ds_tr, idx_ref)
ds_h0 = Subset(ds_tr, idx_h0)
ds_h1 = ds.get_subset('test')
dl = partial(DataLoader, shuffle=True, batch_size=100, collate_fn=ds.collate, num_workers=2)
dl_ref, dl_h0, dl_h1 = dl(ds_ref), dl(ds_h0), dl(ds_h1)
```

## Detect drift

### MMD detector on transformer embeddings

First we embed instances using a pretrained transformer model and detect data drift using the [MMD detector](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html) on the embeddings.

Helper functions:

```{python}
from typing import List


def update_flat_list(x: List[list]):
    return [item for sublist in x for item in sublist]


def accumulate_sample(dataloader: DataLoader, sample_size: int, stars: int = None):
    """ Create batches of data from dataloaders. """
    batch_count, stars_count = 0, 0
    x_out, y_out, meta_out = [], [], []
    for x, y, meta in dataloader:
        y, meta = y.numpy(), meta.numpy()
        if isinstance(stars, int):
            idx_stars = np.where(y == stars)[0]
            y, meta = y[idx_stars], meta[idx_stars]
            x = tuple([x[idx] for idx in idx_stars])
        n_batch = y.shape[0]
        idx = min(sample_size - batch_count, n_batch)
        batch_count += n_batch
        x_out += [x[:idx]]
        y_out += [y[:idx]]
        meta_out += [meta[:idx]]
        if batch_count >= sample_size:
            break
    x_out = update_flat_list(x_out)
    y_out = np.concatenate(y_out, axis=0)
    meta_out = np.concatenate(meta_out, axis=0)
    return x_out, y_out, meta_out
```

Define the transformer embedding preprocessing step:

```{python}
#| scrolled: true
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.models.pytorch import TransformerEmbedding
from functools import partial
from transformers import AutoTokenizer

emb_type = 'hidden_state'  # pooler_output, last_hidden_state or hidden_state
# layers to extract hidden states from for the embedding used in drift detection
# only relevant for emb_type = 'hidden_state'
n_layers = 8
layers = [-_ for _ in range(1, n_layers + 1)]
max_len = 100  # max length for the tokenizer

model_name = 'bert-base-cased'  # a model supported by the transformers library
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding = TransformerEmbedding(model_name, emb_type, layers).to(device).eval()
preprocess_fn = partial(preprocess_drift, model=embedding, tokenizer=tokenizer, max_len=max_len, batch_size=32)
```

Define a function which will for a specified number of iterations (`n_sample`):
- Configure the `MMDDrift` detector with a new reference data sample
- Detect drift on the H0 and H1 splits

```{python}
labels = ['No!', 'Yes!']


def print_preds(preds: dict, preds_name: str) -> None:
    print(preds_name)
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print(f'p-value: {preds["data"]["p_val"]:.3f}')
    print('') 
    

def make_predictions(ref_size: int, test_size: int, n_sample: int, stars_h1: int = 4) -> None:
    """ Create drift MMD detector, init, sample data and make predictions. """
    for _ in range(n_sample):
        # sample data
        x_ref, y_ref, meta_ref = accumulate_sample(dl_ref, ref_size)
        x_h0, y_h0, meta_h0 = accumulate_sample(dl_h0, test_size)
        x_h1, y_h1, meta_h1 = accumulate_sample(dl_h1, test_size, stars=stars_h1)
        # init and run detector
        dd = MMDDrift(x_ref, backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn, n_permutations=1000)
        preds_h0 = dd.predict(x_h0)
        preds_h1 = dd.predict(x_h1)
        print_preds(preds_h0, 'H0')
        print_preds(preds_h1, 'H1')
```

```{python}
#| scrolled: false
make_predictions(ref_size=1000, test_size=1000, n_sample=2, stars_h1=4)
```

### Classifier drift detector

Now we will use the [ClassifierDrift detector](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/classifierdrift.html) which uses a binary classification model to try and distinguish the reference from the test (H0 or H1) data. Drift is then detected on the difference between the prediction distributions on out-of-fold reference vs. test instances using a Kolmogorov-Smirnov 2 sample test on the prediction probabilities or via a binomial test on the binarized predictions. We use a pretrained transformer model but freeze its weights and only train the head which consists of 2 dense layers with a leaky ReLU non-linearity:

```{python}
import torch.nn as nn
from transformers import DistilBertModel

model_name = 'distilbert-base-uncased'

class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm = DistilBertModel.from_pretrained(model_name)
        for param in self.lm.parameters():  # freeze language model weights
            param.requires_grad = False
        self.head = nn.Sequential(nn.Linear(768, 512), nn.LeakyReLU(.1), nn.Linear(512, 2))
    
    def forward(self, tokens) -> torch.Tensor:
        h = self.lm(**tokens).last_hidden_state
        h = nn.MaxPool1d(kernel_size=100)(h.permute(0, 2, 1)).squeeze(-1)
        return self.head(h)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Classifier()
```

```{python}
from alibi_detect.cd import ClassifierDrift
from alibi_detect.utils.prediction import tokenize_transformer


def make_predictions(model, backend: str, ref_size: int, test_size: int, n_sample: int, stars_h1: int = 4) -> None:
    """ Create drift Classifier detector, init, sample data and make predictions. """
    
    # batch_fn tokenizes each batch of instances of the reference and test set during training
    b = 'pt' if backend == 'pytorch' else 'tf'
    batch_fn = partial(tokenize_transformer, tokenizer=tokenizer, max_len=max_len, backend=b)
    
    for _ in range(n_sample):
        # sample data
        x_ref, y_ref, meta_ref = accumulate_sample(dl_ref, ref_size)
        x_h0, y_h0, meta_h0 = accumulate_sample(dl_h0, test_size)
        x_h1, y_h1, meta_h1 = accumulate_sample(dl_h1, test_size, stars=stars_h1)
        # init and run detector
        # since our classifier returns logits, we set preds_type to 'logits'
        # n_folds determines the number of folds used for cross-validation, this makes sure all 
        #   test data is used but only out-of-fold predictions taken into account for the drift detection
        #   alternatively we can set train_size to a fraction between 0 and 1 and not apply cross-validation
        # epochs specifies how many epochs the classifier will be trained for each sample or fold
        # preprocess_batch_fn is applied to each batch of instances and translates the text into tokens
        dd = ClassifierDrift(x_ref, model, backend=backend, p_val=.05, preds_type='logits', 
                             n_folds=3, epochs=2, preprocess_batch_fn=batch_fn, train_size=None)
        preds_h0 = dd.predict(x_h0)
        preds_h1 = dd.predict(x_h1)
        print_preds(preds_h0, 'H0')
        print_preds(preds_h1, 'H1')
```

```{python}
#| scrolled: true
make_predictions(model, 'pytorch', ref_size=1000, test_size=1000, n_sample=2, stars_h1=4)
```

### TensorFlow drift detector

We can do the same using TensorFlow instead of PyTorch as backend. We first define the classifier again and then simply run the detector:

```{python}
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, MaxPool1D
from transformers import TFDistilBertModel

class ClassifierTF(tf.keras.Model):
    def __init__(self) -> None:
        super(ClassifierTF, self).__init__()
        self.lm = TFDistilBertModel.from_pretrained(model_name)
        self.lm.trainable = False  # freeze language model weights
        self.head = tf.keras.Sequential([Dense(512), LeakyReLU(alpha=.1), Dense(2)])
    
    def call(self, tokens) -> tf.Tensor:
        h = self.lm(**tokens).last_hidden_state
        h = tf.squeeze(MaxPool1D(pool_size=100)(h), axis=1)
        return self.head(h)
    
    @classmethod
    def from_config(cls, config):  # not needed for sequential/functional API models
        return cls(**config)

model = ClassifierTF()
```

```{python}
#| scrolled: false
make_predictions(model, 'tensorflow', ref_size=1000, test_size=1000, n_sample=2, stars_h1=4)
```

