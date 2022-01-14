# Detector Configuration Files

For advanced use cases, Alibi Detect features powerful configuration file based functionality. As shown below,
**Drift detectors** can be specified with a configuration file named `config.toml` (adversarial and outlier 
detectors coming soon!), which can then be passed to `load_detector`:

````{panels}
:column: p-1 m-1
:card: shadow

**Standard instantiation**
^^^

```python
import numpy as np
from alibi_detect.cd import MMDDrift

x_ref = np.load('detector_directory/x_ref.npy')
detector = MMDDrift(x_ref, p_val=0.05)
```

TODO - custom styling for code-blocks and panels

---

**Config-driven instantiation**
^^^

<p class="codeblock-label">config.toml</p>

```toml
name = "MMDDrift"
x_ref = "x_ref.npy"
p_val = 0.05
```

```python
from alibi_detect.utils.saving import load_detector
filepath = 'detector_directory/'
detector = load_detector(filepath)
```

````

Compared to *standard instantiation*, config-driven instantiation has a number of advantages:

- **Human readable**: The `config.toml` files are human-readable, providing a readily accessible record of 
previously created detectors.
- **Flexible artefact specification**: Artefacts such as datasets and models can be specified as locally serialized
objects, or as runtime registered objects (see [Specifying complex fields](complex_fields)). Multiple detectors can 
share the same artefacts, and they can be easily swapped.
- **Editable**: Config files written by the `save_detector` function can be edited prior to reloading.
- **Inbuilt validation**: The `load_detector` function uses [pydantic](https://pydantic-docs.helpmanual.io/) to validate
detector configurations, improving reliability.

In what follows, the Alibi Detect config files are explored in some detail. To get a general idea 
of the expected layout of a config file, readers can also skip ahead to [Example config files](examples) for examples of
config files for some common use cases. Alternatively, to obtain a fully populated config file for reference, users
can run one of the [example notebooks](../cd/examples.md) and generate a config file by passing an instantiated 
detector to `save_detector()`.

## Configuration file layout

All detector configuration files follow a consistent layout, simplifying the process of writing simple config files
by hand. For example, a [KSDrift](../api/alibi_detect.cd.ks.rst) detector with a 
[dill](https://github.com/uqfoundation/dill) serialized function to preprocess reference and test data can be specified 
as:

````{tabbed} Config-driven instantiation

<p class="codeblock-label">config.toml</p>

```toml
name = "KSDrift"
x_ref = "x_ref.npy"
p_val = 0.05
preprocess_fn = "function.dill"
```

```python
from alibi_detect.utils.saving import load_detector
detector = load_detector('detector_directory/')
```
````

````{tabbed} Standard instantiation

```python
import numpy as np
from alibi_detect.cd import KSDrift

x_ref = np.load('detector_directory/x_ref.npy')
preprocess_fn = dill.load('detector_directory/function.dill')
detector = MMDDrift(x_ref, p_val=0.05, preprocess_fn=preprocess_fn)
```
````

The `name` field should always be the name of the detector, for example `KSDrift` or `SpotTheDiffDrift`. 
The following fields are the args/kwargs to pass to the detector (see the drift detector 
[api docs](../api/alibi_detect.cd.rst) for a full list of permissible args/kwargs for each detector). All config fields
follow this convention, however as discussed in [Specifying complex fields](complex_fields), some fields can be 
more complex than others. 

```{note}
In the  above example `config.toml`, `x_ref` and `preprocess_fn` are stored in `detector_directory/`, but this directory
isn't included in the config file. This is because in the config file, **relative directories are relative to the 
location of the config.toml file** (absolute filepaths can also be used). 
```

(complex_fields)=
## Specifying complex fields

When specifying a detector via a `config.toml` file, the locally stored reference data `x_ref` must be specified. 
In addition, many detectors also require (or allow) additional artefacts, such as kernels, functions and models. 
Depending on their type, artefacts can be specified in `config.toml` in a number of ways: 

- **Local files**: Simple functions and/or models can be specified as locally stored 
[dill](https://dill.readthedocs.io/en/latest/dill.html) files, whilst data arrays can be specified as locally stored
numpy [npy](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html) files.

- **Function/object registry**: As discussed in [Registering artefacts](registering_artefacts), functions and other 
objects defined at runtime can be registered using `alibi_detect.utils.registry`, allowing them to be specified 
in the config file without having to serialise them. For convenience a number of Alibi Detect functions such as 
[preprocess_drift](../api/alibi_detect.cd.tensorflow.preprocess.rst) are also pre-registered. 

- **Dictionaries**: More complex artefacts are specified via nested dictionaries with a `src` field and 
additional option/setting fields, which are sometimes further nested dictionaries themselves. See 
[Artefact dictionaries](dictionaries) for further details.

The following table shows the allowable formats for all artefacts that can be specified in a config file.

|Field                     |.npy file  |.dill file  |Registry|Dictionary| 
|:-------------------------|:---------:|:----------:|:------:|:--------:|
|`x_ref`                   |✔          |            |        |          |
|`kernel`                  |           |✔           |✔       |          |
|`optimizer`               |           |✔           |✔       |          |
|`reg_loss_fn`             |           |✔           |✔       |          |
|`preprocess_fn`           |           |✔           |✔       |✔         |
|`model`                   |           |            |✔       |✔         |
|`preprocess_fn.model`     |           |            |✔       |✔         |
|`preprocess_fn.embedding` |           |            |✔       |✔         |
|`preprocess_fn.tokenizer` |           |            |✔       |✔         |

TODO: More fields to add in table. 

(registering_artefacts)=
### Registering artefacts

Custom artefacts defined in Python code may be specified in the config file without the need to serialise them, 
by first adding them to the Alibi Detect artifact registry using the [registry](../api/alibi_detect.utils.registry.rst) 
submodule. This submodule harnesses the [catalogue](https://github.com/explosion/catalogue) library to allow functions 
to be registered with a decorator syntax:

````{panels}
:column: p-1 m-1
:card: shadow

**Registering a function**
^^^

```python
import numpy as np
from alibi_detect.utils.registry import registry

@registry.register('my_function.v1')
def my_function(x: np.ndarray) -> np.ndarray:
    "A custom function to normalise input data."
    return (x - x.mean()) / x.std()
```

---

**Specifying in a config.toml**
^^^

<p class="codeblock-label">config.toml</p>

```toml
name = "MMDDrift"
x_ref = "x_ref.npy"
preprocess_fn = "@my_function.v1"
```

````

Once the custom function has been registered, it can be specified in `config.toml` files via its reference string
(with `@` prepended), for example `"@my_function.v1"` in this case. Other objects, such as custom tensorflow or 
pytorch models, can also be registered by using the `register` function directly. For example, to register a tensorflow 
encoder model:

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
from alibi_detect.utils.registry import registry

encoder_net = Sequential(
  [
      InputLayer(input_shape=(32, 32, 3)),
      Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
      Flatten(),
      Dense(32,)
  ]
)
registry.register("my_encoder.v1", func=encoder_net)
```

#### Examining registered artefacts

A registered object's metadata can be obtained with `registry.find()`, and all currently registered 
objects can be listed with `registry.get_all()`. For example, `registry.find("my_function.v1")` returns the following:

```pycon
{'module': '__main__', 'file': 'test.py', 'line_no': 3, 'docstring': 'A custom function to normalise input data.'}
```

#### Pre-registered utility functions/objects
 
For convenience, Alibi Detect also pre-registers a number of commonly used utility functions and objects.

| Function/Class                                                       | Registry reference*                           | Tensorflow | Pytorch | 
|:---------------------------------------------------------------------|:----------------------------------------------|------------|---------|
| [preprocess_drift](../api/alibi_detect.cd.tensorflow.preprocess.rst) | `'@cd.[backend].preprocess.preprocess_drift'` | ✔           | ✔       |
| [GaussianRBF](../api/alibi_detect.utils.tensorflow.kernels.rst)      | `'@utils.[backend].kernels.GaussianRBF'`      | ✔           | ✔       |
| [TFDataset](../api/alibi_detect.utils.tensorflow.data.rst)           | `'@utils.tensorflow.data.TFDataset'`          | ✔           |         |

**For backend-specific functions/classes, [backend] should be replaced the desired backend e.g. `tensorflow` or `pytorch`.*

These can be used in `config.toml` files without the need to explicitly import `alibi_detect.utils.registry`. 
Of particular importance are the `preprocess_drift` utility functions, which allows models, tokenizers and embeddings
to be easily specified for preprocessing, as demonstrated in the [IMDB example](imdb_example). 


(dictionaries)=
### Artefact dictionaries

More complex artefacts can be specified with a dictionary, which might itself contain fields referencing other 
artefacts. TOML files are highly flexible when it comes to specifying dictionaries; each section in 
a TOML file, demarcated by section headers enclosed in [ ] brackets, is actually a Python dictionary (when the file is 
parsed). For example, the `preprocess_fn` can be specified as a dictionary with:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[preprocess_fn]
src = "function.dill"
```

Alternatively, the following are also equivalent:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
preprocess_fn = {src="function.dill"}
```

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
preprocess_fn.src = "function.dill"
```

In the following sections, the layouts of the various possible artefact dictionaries are documented.

#### Preprocessing function

Simple `preprocess_fn`'s can be specified directly as a serialized dill file or via a function registry. If additional 
arguments are to be passed at call time, the `preprocess_fn` can instead be specified as a dictionary:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[preprocess_fn]
src = "function.dill"
kwargs = {kwarg1="a", kwarg2=1.5, kwarg3=true}
```

In this case, the `src` field can be a locally stored `.dill` file or a function registry, and `kwargs` is a 
dictionary of keyword arguments to be passed to the function. As noted previously, there is considerable flexibility 
with regard to how these fields are specified. For example, the following is equivalent:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
preprocess_fn.src = "function.dill"

[preprocess_fn.kwargs]
kwarg1 = "a"
kwarg2 = 1.5
kwarg3 = true
```

A special (but common!) case for `preprocess_fn` is to use the Alibi Detect utility function 
[preprocess_drift](../api/alibi_detect.cd.tensorflow.preprocess.rst) (or its pytorch equivalent). This is a utility 
function to incorporate a user-supplied model, batch preprocessing function, and/or tokenizer into the preprocessing 
step of a drift detector. It is pre-registered, so can be specified with 
`src="@cd.tensorflow.preprocess.preprocess_drift"`, and the rest of the fields in `preprocess_fn` are then the 
args/kwargs to pass to `preprocess_drift`. For example: 

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[preprocess_fn]
src = "@cd.tensorflow.preprocess.preprocess_drift"
batch_size = 32
preprocess_batch_fn = "batch_fn.dill"
```

#### Models 

A `model` dictionary may be used to specify a model within a `preprocess_fn`, or a model for detectors such as the
[ClassifierDrift](../api/alibi_detect.cd.classifier.rst) detector. The possible fields are:

| Field      | Description                                                                                                                                                                                                                                                                     | Default (if optional) |
|:-----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|
| src        | Filepath to directory storing the model (relative to the `config.toml` file, or absolute).                                                                                                                                                                                      |                       |
| type       | The type of model to be loaded. Options:<br/> - `"custom"`: A generic custom model. <br/> - `"HiddenOutput"`: An Alibi Detect [HiddenOutput](../api/alibi_detect.cd.tensorflow.rst) model. <br/> - `"UAE"`: An Alibi Detect [UAE](../api/alibi_detect.cd.tensorflow.rst) model. | `"custom"`            |
| custom_obj | Dictionary of custom objects. Passed to the tensorflow [load_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model) function.                                                                                                                            | `None`                |

The `src` should refer to a directory containing a TensorFlow model stored in the 
[Keras H5 format](https://www.tensorflow.org/guide/keras/save_and_serialize#keras_h5_format) 
(more model formats will be supported in the future). Below is a simple example config to load a generic TensorFlow
model stored in `model/`.

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[model]
type = "custom"
src = "model"
```

#### Embedding

As demonstrated in [Text drift detection on IMDB movie reviews](../examples/cd_text_imdb.ipynb), pre-trained embeddings
can be extracted from [HuggingFace’s transformer package](https://github.com/huggingface/transformers) for use as a 
preprocessing step. For this purpose, models specified in the `embedding` field will be passed through the Alibi Detect 
[TransformerEmbedding](../api/alibi_detect.models.tensorflow.embedding.rst) function (or its pytorch equivalent).

| Field  | Description                                                                                                                                                     | Default (if optional) |
|:-------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|
| src    | Model name e.g. `"bert-base-cased"`, or a filepath to directory storing the model to extract embeddings from (relative to the `config.toml` file, or absolute). |                       |
| type   | The type of embedding to be loaded. See `embedding_type` in [TransformerEmbedding](../api/alibi_detect.models.tensorflow.embedding.rst).                        |                       |
| layers | List specifying the hidden layers to be used to extract the embedding.                                                                                          | `None`                |

Example:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[model]
src = "bert-base-cased"
type = "hidden_state"
layers = [-1, -2, -3, -4, -5, -6, -7, -8]
```

#### Tokenizers

Pre-trained tokenizers from [HuggingFace's tokenizers package](https://github.com/huggingface/tokenizers) may be 
specified via the `tokenizer` field.

| Field  | Description                                                                                                                                                                                | Default (if optional) |
|:-------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|
| src    | Model name e.g. `"bert-base-cased"`, or a filepath to directory storing the tokenizer model (relative to the `config.toml` file, or absolute).                                             |                       |
| kwargs | Dictionary of keyword arguments to pass to [AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained). | `{}`                  |

#### Kernels

| Field     | Description                                                       | Default (if optional) |
|:----------|:------------------------------------------------------------------|:----------------------|
| src       | `"@utils.tensorflow.kernels.GaussianRBF"`                         |                       |
| sigma     |                                                                   |                       |
| trainable |                                                                   |                       |
| kwargs    | Dictionary of additional keyword arguments to pass to the kernel. | `{}`                  |

# TODO - does it make sense to have defaults? Related to None issue
# TODO - Better way to do kwargs? i.e. include sigma etc in kwargs? TypedDict etc...

Example:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[]
```

#### Optimizers

Dictionary in format used by `tf.keras.optimizers.serialize`
Example:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[]
```

(examples)=
## Example config files

(imdb_example)=
### Drift detection on text data
from the IMDB example

<p class="codeblock-label">config.toml</p>

```toml
x_ref = "x_ref.npy"
name = "LSDDDrift"

[preprocess_fn]
src = "@cd.tensorflow.preprocess.preprocess_drift"
batch_size = 32
max_len = 100
tokenizer.src = "tokenizer/"

[preprocess_fn.model]
type = "UAE"
src = "model/"

[preprocess_fn.embedding]
src = "embedding/"
type = "hidden_state"
layers = [-1, -2, -3, -4, -5, -6, -7, -8]
```

### A classifier based drift detector

From the clf cifar10 example

<p class="codeblock-label">config.toml</p>

```toml
name = "ClassifierDrift"
x_ref = "x_ref.npy"
input_shape = [ 32, 32, 3,]
batch_size = 32
epochs = 5

[model]
type = "custom"
src = "model"

[optimizer]
class_name = "Adam"

[optimizer.config]
name = "Adam"
learning_rate = 0.001
```

## Advanced usage


### Validating config files

Talk about pydantic. The unresolved and resolved config dict is validated with pydantic...

Public function for doing this...

### Detector specification schemas

Can/should implement a public facing api to fetch the pydantic config schema's in json format. Discuss here...


## Limitations

- When loading a config file generated with `save_detector`, a warning will be issued if the runtime alibi-detect version is 
different from the version used to save the detector. **It is highly recommended to use the same 
alibi-detect, Python and dependency versions as were used to save the detector to avoid potential 
bugs and incompatibilities**.

- Save/load functionality is currently only implemented for offline drift detectors, using the tensorflow backend and/or tensorflow models. It will be updated to online detectors and pytorch detectors/models in the future.

