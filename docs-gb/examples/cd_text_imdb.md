---
title: Text drift detection on IMDB movie reviews
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---


## Method

We detect drift on text data using both the [Maximum Mean Discrepancy](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html) and [Kolmogorov-Smirnov (K-S)](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html) detectors. In this example notebook we will focus on detecting covariate shift $\Delta p(x)$ as detecting predicted label distribution drift does not differ from other modalities (check [K-S](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_ks_cifar10.html#BBSDs) and [MMD](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_mmd_cifar10.html#BBSDs) drift on CIFAR-10).

It becomes however a little bit more involved when we want to pick up input data drift $\Delta p(x)$. When we deal with tabular or image data, we can either directly apply the two sample hypothesis test on the input or do the test after a preprocessing step with for instance a randomly initialized encoder as proposed in [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953) (they call it an Untrained AutoEncoder or *UAE*). It is not as straightforward when dealing with text, both in string or tokenized format as they don't directly represent the semantics of the input.

As a result, we extract (contextual) embeddings for the text and detect drift on those. This procedure has a significant impact on the type of drift we detect. Strictly speaking we are not detecting $\Delta p(x)$ anymore since the whole training procedure (objective function, training data etc) for the (pre)trained embeddings has an impact on the embeddings we extract.

The library contains functionality to leverage pre-trained embeddings from [HuggingFace's transformer package](https://github.com/huggingface/transformers) but also allows you to easily use your own embeddings of choice. Both options are illustrated with examples in this notebook.


<div class="alert alert-info">
Note

As is done in this example, it is recommended to pass text data to detectors as a list of strings (`List[str]`). This allows for seamless integration with HuggingFace's transformers library.

One exception to the above is when custom embeddings are used. Here, it is important to ensure that the data is passed to the custom embedding model in a compatible format. In [the final example](#Train-embeddings-from-scratch), a `preprocess_batch_fn` is defined in order to convert `list`'s to the `np.ndarray`'s expected by the custom TensorFlow embedding.
    
</div>

## Backend

The method works with both the **PyTorch** and **TensorFlow** frameworks for the statistical tests and preprocessing steps. Alibi Detect does however not install PyTorch for you. 
Check the [PyTorch docs](https://pytorch.org/) how to do this.

## Dataset

Binary sentiment classification [dataset](https://ai.stanford.edu/~amaas/data/sentiment/) containing $25,000$ movie reviews for training and $25,000$ for testing. Install the `nlp` library to fetch the dataset:


```{python}
!pip install nlp
```

```{python}
import nlp
import numpy as np
import os
import tensorflow as tf
from transformers import AutoTokenizer
from alibi_detect.cd import KSDrift, MMDDrift
from alibi_detect.saving import save_detector, load_detector
```

### Load tokenizer

```{python}
#| scrolled: true
model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Load data

```{python}
def load_dataset(dataset: str, split: str = 'test'):
    data = nlp.load_dataset(dataset)
    X, y = [], []
    for x in data[split]:
        X.append(x['text'])
        y.append(x['label'])
    X = np.array(X)
    y = np.array(y)
    return X, y
```

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
#| scrolled: true
X, y = load_dataset('imdb', split='train')
print(X.shape, y.shape)
```

Let's take a look at respectively a negative and positive review:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
labels = ['Negative', 'Positive']
print(labels[y[-1]])
print(X[-1])
```

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
print(labels[y[2]])
print(X[2])
```

We split the original test set in a reference dataset and a dataset which should not be rejected under the *H0* of the statistical test. We also create imbalanced datasets and inject selected words in the reference set.

```{python}
def random_sample(X: np.ndarray, y: np.ndarray, proba_zero: float, n: int):
    if len(y.shape) == 1:
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
    else:
        idx_0 = np.where(y[:, 0] == 1)[0]
        idx_1 = np.where(y[:, 1] == 1)[0]
    n_0, n_1 = int(n * proba_zero), int(n * (1 - proba_zero))
    idx_0_out = np.random.choice(idx_0, n_0, replace=False)
    idx_1_out = np.random.choice(idx_1, n_1, replace=False)
    X_out = np.concatenate([X[idx_0_out], X[idx_1_out]])
    y_out = np.concatenate([y[idx_0_out], y[idx_1_out]])
    return X_out.tolist(), y_out.tolist()


def padding_last(x: np.ndarray, seq_len: int) -> np.ndarray:
    try:  # try not to replace padding token
        last_token = np.where(x == 0)[0][0]
    except:  # no padding
        last_token = seq_len - 1
    return 1, last_token


def padding_first(x: np.ndarray, seq_len: int) -> np.ndarray:
    try:  # try not to replace padding token
        first_token = np.where(x == 0)[0][-1] + 2
    except:  # no padding
        first_token = 0
    return first_token, seq_len - 1


def inject_word(token: int, X: np.ndarray, perc_chg: float, padding: str = 'last'):
    seq_len = X.shape[1]
    n_chg = int(perc_chg * .01 * seq_len)
    X_cp = X.copy()
    for _ in range(X.shape[0]):
        if padding == 'last':
            first_token, last_token = padding_last(X_cp[_, :], seq_len)
        else:
            first_token, last_token = padding_first(X_cp[_, :], seq_len)
        if last_token <= n_chg:
            choice_len = seq_len
        else:
            choice_len = last_token
        idx = np.random.choice(np.arange(first_token, choice_len), n_chg, replace=False)
        X_cp[_, idx] = token
    return X_cp.tolist()
```

Reference, *H0* and imbalanced data:

```{python}
# proba_zero = fraction with label 0 (=negative sentiment)
n_sample = 1000
X_ref = random_sample(X, y, proba_zero=.5, n=n_sample)[0]
X_h0 = random_sample(X, y, proba_zero=.5, n=n_sample)[0]
n_imb = [.1, .9]
X_imb = {_: random_sample(X, y, proba_zero=_, n=n_sample)[0] for _ in n_imb}
```

Inject words in reference data:

```{python}
words = ['fantastic', 'good', 'bad', 'horrible']
perc_chg = [1., 5.]  # % of tokens to change in an instance

words_tf = tokenizer(words)['input_ids']
words_tf = [token[1:-1][0] for token in words_tf]
max_len = 100
tokens = tokenizer(X_ref, pad_to_max_length=True, 
                   max_length=max_len, return_tensors='tf')
X_word = {}
for i, w in enumerate(words_tf):
    X_word[words[i]] = {}
    for p in perc_chg:
        x = inject_word(w, tokens['input_ids'].numpy(), p)
        dec = tokenizer.batch_decode(x, **dict(skip_special_tokens=True))
        X_word[words[i]][p] = dec
```

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
tokens['input_ids']
```

## Preprocessing

First we need to specify the type of embedding we want to extract from the BERT model. We can extract embeddings from the ...

- **pooler_output**: Last layer hidden-state of the first token of the sequence (classification token; CLS) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pre-training. **Note**: this output is usually not a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.

- **last_hidden_state**: Sequence of hidden states at the output of the last layer of the model, averaged over the tokens.

- **hidden_state**: Hidden states of the model at the output of each layer, averaged over the tokens.

- **hidden_state_cls**: See *hidden_state* but use the CLS token output.

If *hidden_state* or *hidden_state_cls* is used as embedding type, you also need to pass the layer numbers used to extract the embedding from. As an example we extract embeddings from the last 8 hidden states.

```{python}
#| scrolled: true
from alibi_detect.models.tensorflow import TransformerEmbedding

emb_type = 'hidden_state'
n_layers = 8
layers = [-_ for _ in range(1, n_layers + 1)]

embedding = TransformerEmbedding(model_name, emb_type, layers)
```

Let's check what an embedding looks like:

```{python}
#| scrolled: false
tokens = tokenizer(list(X[:5]), pad_to_max_length=True, 
                   max_length=max_len, return_tensors='tf')
x_emb = embedding(tokens)
print(x_emb.shape)
```

So the BERT model's embedding space used by the drift detector consists of a $768$-dimensional vector for each instance. We will therefore first apply a dimensionality reduction step with an Untrained AutoEncoder (*UAE*) before conducting the statistical hypothesis test. We use the embedding model as the input for the UAE which then projects the embedding on a lower dimensional space.

```{python}
tf.random.set_seed(0)
```

```{python}
from alibi_detect.cd.tensorflow import UAE

enc_dim = 32
shape = (x_emb.shape[1],)

uae = UAE(input_layer=embedding, shape=shape, enc_dim=enc_dim)
```

Let's test this again:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
emb_uae = uae(tokens)
print(emb_uae.shape)
```

## K-S detector

### Initialize

We proceed to initialize the drift detector. From here on the detector works the same as for other modalities such as images. Please check the [images](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_ks_cifar10.html) example or the [K-S detector documentation](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/ksdrift.html) for more information about each of the possible parameters.

```{python}
#| scrolled: true
from functools import partial
from alibi_detect.cd.tensorflow import preprocess_drift

# define preprocessing function
preprocess_fn = partial(preprocess_drift, model=uae, tokenizer=tokenizer, 
                        max_len=max_len, batch_size=32)

# initialize detector
cd = KSDrift(X_ref, p_val=.05, preprocess_fn=preprocess_fn, input_shape=(max_len,))

# we can also save/load an initialised detector
filepath = 'my_path'  # change to directory where detector is saved
save_detector(cd, filepath)
cd = load_detector(filepath)
```

### Detect drift

Let’s first check if drift occurs on a similar sample from the training set as the reference data.

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
preds_h0 = cd.predict(X_h0)
labels = ['No!', 'Yes!']
print('Drift? {}'.format(labels[preds_h0['data']['is_drift']]))
print('p-value: {}'.format(preds_h0['data']['p_val']))
```

Detect drift on imbalanced and perturbed datasets:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
for k, v in X_imb.items():
    preds = cd.predict(v)
    print('% negative sentiment {}'.format(k * 100))
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print('p-value: {}'.format(preds['data']['p_val']))
    print('')
```

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
#| scrolled: false
for w, probas in X_word.items():
    for p, v in probas.items():
        preds = cd.predict(v)
        print('Word: {} -- % perturbed: {}'.format(w, p))
        print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        print('p-value: {}'.format(preds['data']['p_val']))
        print('')
```

## MMD TensorFlow detector

### Initialize

Again check the [images](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_mmd_cifar10.html) example or the [MMD detector documentation](https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/mmddrift.html) for more information about each of the possible parameters.

```{python}
cd = MMDDrift(X_ref, p_val=.05, preprocess_fn=preprocess_fn, 
              n_permutations=100, input_shape=(max_len,))
```

### Detect drift

*H0*:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
preds_h0 = cd.predict(X_h0)
labels = ['No!', 'Yes!']
print('Drift? {}'.format(labels[preds_h0['data']['is_drift']]))
print('p-value: {}'.format(preds_h0['data']['p_val']))
```

Imbalanced data:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
for k, v in X_imb.items():
    preds = cd.predict(v)
    print('% negative sentiment {}'.format(k * 100))
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print('p-value: {}'.format(preds['data']['p_val']))
    print('')
```

Perturbed data:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
#| scrolled: false
for w, probas in X_word.items():
    for p, v in probas.items():
        preds = cd.predict(v)
        print('Word: {} -- % perturbed: {}'.format(w, p))
        print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        print('p-value: {}'.format(preds['data']['p_val']))
        print('')
```

## MMD PyTorch detector

### Initialize

We can run the same detector with *PyTorch* backend for both the preprocessing step and MMD implementation:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
import torch
import torch.nn as nn

# set random seed and device
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```

```{python}
#| scrolled: true
from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.models.pytorch import TransformerEmbedding
from alibi_detect.cd.pytorch import UAE

# Embedding model
embedding_pt = TransformerEmbedding(model_name, emb_type, layers)

# PyTorch untrained autoencoder
uae = UAE(input_layer=embedding_pt, shape=shape, enc_dim=enc_dim)
model = uae.to(device).eval()

# define preprocessing function
preprocess_fn = partial(preprocess_drift, model=model, tokenizer=tokenizer, 
                        max_len=max_len, batch_size=32, device=device)

# initialise drift detector
cd = MMDDrift(X_ref, backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn, 
              n_permutations=100, input_shape=(max_len,))
```

### Detect drift

*H0*:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
preds_h0 = cd.predict(X_h0)
labels = ['No!', 'Yes!']
print('Drift? {}'.format(labels[preds_h0['data']['is_drift']]))
print('p-value: {}'.format(preds_h0['data']['p_val']))
```

Imbalanced data:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
for k, v in X_imb.items():
    preds = cd.predict(v)
    print('% negative sentiment {}'.format(k * 100))
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print('p-value: {}'.format(preds['data']['p_val']))
    print('')
```

Perturbed data:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
for w, probas in X_word.items():
    for p, v in probas.items():
        preds = cd.predict(v)
        print('Word: {} -- % perturbed: {}'.format(w, p))
        print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        print('p-value: {}'.format(preds['data']['p_val']))
        print('')
```

## Train embeddings from scratch

So far we used pre-trained embeddings from a BERT model. We can however also use embeddings from a model trained from scratch. First we define and train a simple classification model consisting of an embedding and LSTM layer in *TensorFlow*.

### Load data and train model

```{python}
from tensorflow.keras.datasets import imdb, reuters
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical

INDEX_FROM = 3
NUM_WORDS = 10000


def print_sentence(tokenized_sentence: str, id2w: dict):
    print(' '.join(id2w[_] for _ in tokenized_sentence))
    print('')
    print(tokenized_sentence)


def mapping_word_id(data):
    w2id = data.get_word_index()
    w2id = {k: (v + INDEX_FROM) for k, v in w2id.items()}
    w2id["<PAD>"] = 0
    w2id["<START>"] = 1
    w2id["<UNK>"] = 2
    w2id["<UNUSED>"] = 3
    id2w = {v: k for k, v in w2id.items()}
    return w2id, id2w


def get_dataset(dataset: str = 'imdb', max_len: int = 100):
    if dataset == 'imdb':
        data = imdb
    elif dataset == 'reuters':
        data = reuters
    else:
        raise NotImplementedError

    w2id, id2w = mapping_word_id(data)

    (X_train, y_train), (X_test, y_test) = data.load_data(
        num_words=NUM_WORDS, index_from=INDEX_FROM)
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test), (w2id, id2w)


def imdb_model(X: np.ndarray, num_words: int = 100, emb_dim: int = 128,
               lstm_dim: int = 128, output_dim: int = 2) -> tf.keras.Model:
    X = np.array(X)
    inputs = Input(shape=(X.shape[1:]), dtype=tf.float32)
    x = Embedding(num_words, emb_dim)(inputs)
    x = LSTM(lstm_dim, dropout=.5)(x)
    outputs = Dense(output_dim, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
```

Load and tokenize data:

```{python}
#| scrolled: false
(X_train, y_train), (X_test, y_test), (word2token, token2word) = \
    get_dataset(dataset='imdb', max_len=max_len)
```

Let's check out an instance:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
print_sentence(X_train[0], token2word)
```

Define and train a simple model:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
model = imdb_model(X=X_train, num_words=NUM_WORDS, emb_dim=256, lstm_dim=128, output_dim=2)
model.fit(X_train, y_train, batch_size=32, epochs=2, 
          shuffle=True, validation_data=(X_test, y_test))
```

Extract the embedding layer from the trained model and combine with UAE preprocessing step:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
embedding = tf.keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
x_emb = embedding(X_train[:5])
print(x_emb.shape)
```

```{python}
tf.random.set_seed(0)

shape = tuple(x_emb.shape[1:])
uae = UAE(input_layer=embedding, shape=shape, enc_dim=enc_dim)
```

Again, create reference, *H0* and perturbed datasets. Also test against the *Reuters* news topic classification dataset.

```{python}
X_ref, y_ref = random_sample(X_test, y_test, proba_zero=.5, n=n_sample)
X_h0, y_h0 = random_sample(X_test, y_test, proba_zero=.5, n=n_sample)
tokens = [word2token[w] for w in words]
X_word = {}
for i, t in enumerate(tokens):
    X_word[words[i]] = {}
    for p in perc_chg:
        X_word[words[i]][p] = inject_word(t, np.array(X_ref), p, padding='first')
```

```{python}
#| scrolled: true
# load and tokenize Reuters dataset
(X_reut, y_reut), (w2t_reut, t2w_reut) = \
    get_dataset(dataset='reuters', max_len=max_len)[1:]

# sample random instances
idx = np.random.choice(X_reut.shape[0], n_sample, replace=False)
X_ood = X_reut[idx]
```

### Initialize detector and detect drift

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
from alibi_detect.cd.tensorflow import preprocess_drift

# define preprocess_batch_fn to convert list of str's to np.ndarray to be processed by `model`
def convert_list(X: list):
    return np.array(X)

# define preprocessing function
preprocess_fn = partial(preprocess_drift, model=uae, batch_size=128, preprocess_batch_fn=convert_list)

# initialize detector
cd = KSDrift(X_ref, p_val=.05, preprocess_fn=preprocess_fn)
```

*H0*:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
preds_h0 = cd.predict(X_h0)
labels = ['No!', 'Yes!']
print('Drift? {}'.format(labels[preds_h0['data']['is_drift']]))
print('p-value: {}'.format(preds_h0['data']['p_val']))
```

Perturbed data:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
#| scrolled: false
for w, probas in X_word.items():
    for p, v in probas.items():
        preds = cd.predict(v)
        print('Word: {} -- % perturbed: {}'.format(w, p))
        print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        print('p-value: {}'.format(preds['data']['p_val']))
        print('')
```

The detector is not as sensitive as the Transformer-based K-S drift detector. The embeddings trained from scratch only trained on a small dataset and a simple model with cross-entropy loss function for 2 epochs. The pre-trained BERT model on the other hand captures semantics of the data better.

Sample from the Reuters dataset:

```{python}
#| colab: {base_uri: 'https://localhost:8080/'}
preds_ood = cd.predict(X_ood)
labels = ['No!', 'Yes!']
print('Drift? {}'.format(labels[preds_ood['data']['is_drift']]))
print('p-value: {}'.format(preds_ood['data']['p_val']))
```

