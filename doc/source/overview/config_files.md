# Detector Configuration Files

For advanced use cases, Alibi Detect features powerful configuration file based functionality. As shown below,
**Drift detectors** can be specified with a configuration file named `config.toml` (adversarial and outlier 
detectors coming soon!), which can then be passed to `load_detector`:


`````{grid} 2

````{grid-item-card}
:shadow: md
:margin: 1
:padding: 1
:columns: auto


**Standard instantiation**
^^^

```python
import numpy as np
from alibi_detect.cd import MMDDrift

x_ref = np.load('detector_directory/x_ref.npy')
detector = MMDDrift(x_ref, p_val=0.05)
```
````

````{grid-item-card}
:shadow: md
:margin: 1
:padding: 1
:columns: auto

**Config-driven instantiation**
^^^

<p class="codeblock-label">config.toml</p>

```{code-block} toml

name = "MMDDrift"
x_ref = "x_ref.npy"
p_val = 0.05
```

```python
from alibi_detect.utils.loading import load_detector
filepath = 'detector_directory/'
detector = load_detector(filepath)
```
````
`````

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

`````{tab-set}
````{tab-item} Config-driven instantiation

<p class="codeblock-label">config.toml</p>

```toml
name = "KSDrift"
x_ref = "x_ref.npy"
p_val = 0.05
preprocess_fn = "function.dill"
```

```python
from alibi_detect.utils.loading import load_detector
detector = load_detector('detector_directory/')
```
````

````{tab-item} Standard instantiation

```python
import numpy as np
from alibi_detect.cd import KSDrift

x_ref = np.load('detector_directory/x_ref.npy')
preprocess_fn = dill.load('detector_directory/function.dill')
detector = MMDDrift(x_ref, p_val=0.05, preprocess_fn=preprocess_fn)
```
````
`````

The `name` field should always be the name of the detector, for example `KSDrift` or `SpotTheDiffDrift`. 
The following fields are the args/kwargs to pass to the detector (see the drift detector 
[api docs](../api/alibi_detect.cd.rst) for a full list of permissible args/kwargs for each detector). All config fields
follow this convention, however as discussed in [Specifying artefacts](complex_fields), some fields can be 
more complex than others. 

```{note}
In the  above example `config.toml`, `x_ref` and `preprocess_fn` are stored in `detector_directory/`, but this directory
isn't included in the config file. This is because in the config file, **relative directories are relative to the 
location of the config.toml file** (absolute filepaths can also be used). 
```

(complex_fields)=
## Specifying artefacts

When specifying a detector via a `config.toml` file, the locally stored reference data `x_ref` must be specified. 
In addition, many detectors also require (or allow) additional artefacts, such as kernels, functions and models. 
Depending on their type, artefacts can be specified in `config.toml` in a number of ways: 

- **Local files**: Simple functions and/or models can be specified as locally stored 
[dill](https://dill.readthedocs.io/en/latest/dill.html) files, whilst data arrays are specified as locally stored
numpy [npy](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html) files.

- **Function/object registry**: As discussed in [Registering artefacts](registering_artefacts), functions and other 
objects defined at runtime can be registered using `alibi_detect.utils.registry`, allowing them to be specified 
in the config file without having to serialise them. For convenience a number of Alibi Detect functions such as 
[preprocess_drift](../api/alibi_detect.cd.tensorflow.preprocess.rst) are also pre-registered. 

- **Dictionaries**: More complex artefacts are specified via nested dictionaries with a `src` field and 
additional option/setting fields, which are sometimes further nested dictionaries themselves. See 
[Artefact dictionaries](dictionaries) for further details.

The following table shows the allowable formats for all artefacts that can be specified in a config file.

```{table} Possible artefact formats
:name: all-artefacts-table

|Field                     |.npy file  |.dill file  |Registry|Dictionary| 
|:-------------------------|:---------:|:----------:|:------:|:--------:|
|`x_ref`                   |✔          |            |        |          |
|`optimizer`               |           |✔           |✔       |          |
|`reg_loss_fn`             |           |✔           |✔       |          |
|`model`                   |           |            |✔       |✔         |
|`preprocess_fn`           |           |✔           |✔       |✔         |
|`preprocess_fn.model`     |           |            |✔       |✔         |
|`preprocess_fn.embedding` |           |            |✔       |✔         |
|`preprocess_fn.tokenizer` |           |            |✔       |✔         |
|`dataset`                 |           |✔           |✔       |✔         |
|`kernel`                  |           |✔           |✔       |✔         |
|`kernel.proj`             |           |            |✔       |✔         |
|`kernel.kernel_a`         |           |✔           |✔       |✔         |
|`kernel.kernel_b`         |           |✔           |✔       |✔         |
|`initial_diffs`           |✔          |            |        |          |
```

```{note}
When TensorFlow, PyTorch, or scikit-learn `model`'s are specified, the model type should be specified via the `backend` field in the *config.toml*.
For example, if `model` is a TensorFlow model, set `backend='tensorflow'. This is the case even for detectors that don't take a `backend` kwarg.
```

(dictionaries)=
### Artefact dictionaries

The period in the `preprocess_fn.model` field in the {ref}`all-artefacts-table` table indicates that `model` is in 
fact a nested field. In other words, `preprocess_fn` is an [Artefact dictionary](dictionaries), and `model` is a field 
within the `preprocess_fn` dictionary. This layout is required to specify more complex artefacts such as 
`preprocess_fn`, which themselves rely on other artefacts. TOML files are highly flexible when it comes to specifying 
dictionaries; each section in a TOML file, demarcated by section headers enclosed in [ ] brackets, is actually a Python 
dictionary (when the file is parsed). For example, the `preprocess_fn` can be specified as a dictionary with:

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

The above schemas are not strictly necessary in this case, since `preprocess_fn` could simply be specified with
`preprocess_fn = "function.dill"` here. However, the need to specify fields as dictionaries will become clear in the 
following sections, where the layouts of the various possible artefact dictionaries are documented.

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

```{list-table} Model schema
:header-rows: 1
:name: model-table
:widths: auto

* - Field
  - Description
  - Default (if optional)

* - src
  - Filepath to directory storing the model (relative to the `config.toml` file, or absolute). 
  -
  
* - layer
  - Optional index of hidden layer to extract. If not `None`, a 
  [HiddenOutput](../api/alibi_detect.cd.tensorflow.preprocess.rst) model is returned.
  - `None`
  
* - custom_obj
  - Dictionary of custom objects. Passed to the tensorflow 
  [load_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model) function. This can be used to
  pass custom registered functions and classes to a model.
  - `None`
```

The `src` should refer to a directory containing a TensorFlow model stored in the 
[Keras H5 format](https://www.tensorflow.org/guide/keras/save_and_serialize#keras_h5_format) 
(more model formats will be supported in the future). Below is a simple example config to load a generic TensorFlow
model stored in `model/`.

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[model]
src = "model"
```

#### Embedding

As demonstrated in [Text drift detection on IMDB movie reviews](../examples/cd_text_imdb.ipynb), pre-trained embeddings
can be extracted from [HuggingFace’s transformer package](https://github.com/huggingface/transformers) for use as a 
preprocessing step. For this purpose, models specified in the `embedding` field will be passed to the Alibi Detect 
[TransformerEmbedding](../api/alibi_detect.models.tensorflow.embedding.rst) function (or its pytorch equivalent).

```{list-table} Embedding schema
:header-rows: 1
:name: embedding-table
:widths: auto

* - Field
  - Description
  - Default (if optional)

* - src
  - Model name e.g. `"bert-base-cased"`, or a filepath to directory storing the model to extract embeddings from (relative to the `config.toml` file, or absolute).
  -
* - type
  - The type of embedding to be loaded. See `embedding_type` in [TransformerEmbedding](../api/alibi_detect.models.tensorflow.embedding.rst). 
  -
* - layers
  - List specifying the hidden layers to be used to extract the embedding. 
  - `None`
```  

The `embedding` field is set as part of the `preprocess_fn` dictionary e.g. `preprocess_fn.embedding`, when 
`preprocess_fn.src = "@cd.[backend].preprocess.preprocess_drift"`. If a `preprocess_fn.model` is also specified, 
the embedding and model are chained together, providing a further dimension reduction step. 
For example:

```python
model = model(input_layer=embedding, ...)
```

The resulting model is then passed to the [preprocess_drift](https://docs.seldon.io/projects/alibi-detect/en/latest/api/alibi_detect.cd.tensorflow.html?highlight=preprocess_drift#alibi_detect.cd.tensorflow.preprocess_drift)
function's `model` kwarg. If `preprocess_fn.model` is not set, then `preprocess_fn.embedding` is passed to the 
`model` kwarg by itself. 

Example:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[embedding]
src = "bert-base-cased"
type = "hidden_state"
layers = [-1, -2, -3, -4, -5, -6, -7, -8]
```

#### Tokenizers

For use as a preprocessing step on text data, pre-trained tokenizers from 
[HuggingFace's tokenizers package](https://github.com/huggingface/tokenizers) can be 
specified via the `tokenizer` field.

```{list-table} Tokenizer schema
:header-rows: 1
:name: tokenizer-table
:widths: auto

* - Field
  - Description
  - Default (if optional)
* - src
  - Model name e.g. `"bert-base-cased"`, or a filepath to directory storing the tokenizer model (relative to the `config.toml` file, or absolute). 
  -
* - kwargs
  - Dictionary of keyword arguments to pass to [AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained).
  - `{}`
```

#### Kernels

Some detectors such as the [MMDDrift](../api/alibi_detect.cd.mmd.rst) detector make use of kernels, which are 
specified via the `kernel` field. 

```{list-table} Standard kernel schema
:header-rows: 1
:name: kernel-table
:widths: auto

* - Field
  - Description
  - Default (if optional)
* - src
  - Filepath to kernel serialized in a '.dill` file, or reference to a registered kernel object.
  -
* - sigma
  - List of floats to pass as bandwidths to kernel. Only used if 
  [GaussianRBF](https://docs.seldon.io/projects/alibi-detect/en/latest/api/alibi_detect.utils.tensorflow.html) specified.
  - `None`
* - trainable
  - `True`/`False`. Whether or not to track gradients w.r.t. sigma, allowing the kernel to be trained. Only used if 
  [GaussianRBF](https://docs.seldon.io/projects/alibi-detect/en/latest/api/alibi_detect.utils.tensorflow.html) specified. 
  - `False`
* - kwargs
  - Dictionary of additional keyword arguments to pass to the kernel. Only used if kernels other than 
  [GaussianRBF](https://docs.seldon.io/projects/alibi-detect/en/latest/api/alibi_detect.utils.tensorflow.html) specified.  
  - `{}`
```

The default kernel for [MMDDrift](../api/alibi_detect.cd.mmd.rst) and other detectors is the 
[GaussianRBF](https://docs.seldon.io/projects/alibi-detect/en/latest/api/alibi_detect.utils.tensorflow.html) kernel.
This Alibi Detect class is pre-registered (see [Registering artefacts](registering_artefacts)), meaning it can be 
specified with `src = "@utils.tensorflow.kernels.GaussianRBF"` (replace `tensorflow` with `pytorch` for the pytorch
version). If the specified kernel is a `GaussianRBF` kernel, the `sigma` and `trainable` kwargs are passed to it. For all other 
kernels, the generic `kwargs` dict is passed.

The [LearnedKernel](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/learnedkerneldrift.html) detector
requires a [DeepKernel](https://docs.seldon.io/projects/alibi-detect/en/latest/api/alibi_detect.utils.tensorflow.html?highlight=deepkernel#alibi_detect.utils.tensorflow.DeepKernel)
to be passed. This is also specified by the `kernel` field, but with a slightly different schema: 

```{list-table} DeepKernel kernel schema
:header-rows: 1
:name: deepkernel-table
:widths: auto

* - Field
  - Description
  - Default (if optional)
* - `kernel_a`
  -  Kernel to apply to projected inputs. A string referencing a kernel serialized in a `.dill` file, a registered 
  kernel object, or a kernel artefact dictionary (see {ref}`kernel-table`).
  - `"@utils.tensorflow.kernels.GaussianRBF"` <br/>(with `trainable = True`)
* - `kernel_b`
  -  Kernel to apply to raw inputs. A string referencing a kernel serialized in a `.dill` file, a registered kernel 
  object, or a kernel artefact dictionary (see {ref}`kernel-table`).
  - `None`
* - `proj`
  - Projection to be applied to the inputs. Should be a TensorFlow model specified as a registered 
  model, or as an artefact dictionary following the layout in {ref}`model-table`. 
  - 
* - `eps`
  - The proportion (in [0,1]) of weight to assign to `kernel_b`. Specified as a float, or set to `"trainable"`. 
  - `"trainable"`
```

As shown in the example below, when a `DeepKernel` is specified, `kernel_a` and `kernel_b` are themselves kernels, 
and can be specified via the {ref}`kernel-table`.

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[kernel]
eps = 0.01

[kernel.kernel_a]
src = "@utils.tensorflow.kernels.GaussianRBF"
trainable = true

[kernel.kernel_b]
src = "custom_kernel.dill"
sigma = [ 1.2,]
trainable = false

[kernel.proj]
src = "model/"
```

#### Optimizers

Optimizers, required by detectors such as `LearnedKernelDrift` and `ClassifierDrift`, are specified via the
`optimizer` field. For the `tensorflow` backend, `optimizer` should be specified as an artefact dictionary following the
config schema expected by 
[tf.keras.optimizers.deserialize](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/deserialize).
A TensorFlow `Optimizer` config can be generated with the 
[tf.keras.optimizers.serialize](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/serialize) function.

Example:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[optimizer]
class_name = "Adam"

[optimizer.config]
name = "Adam"
learning_rate = 0.01
beta_1 = 0.89
beta_2 = 0.99
epsilon = 1e-7
amsgrad = false
```

(registering_artefacts)=
### Registering artefacts

Custom artefacts defined in Python code may be specified in the config file without the need to serialise them, 
by first adding them to the Alibi Detect artifact registry using the [registry](../api/alibi_detect.utils.registry.rst) 
submodule. This submodule harnesses the [catalogue](https://github.com/explosion/catalogue) library to allow functions 
to be registered with a decorator syntax:


`````{grid} 2

````{grid-item-card}
:shadow: md
:margin: 1
:padding: 1
:columns: auto

**Registering a function**
^^^

```python
import numpy as np
from alibi_detect.utils.registry import registry
from alibi_detect.utils.loading import load_detector

# Register a simple function
@registry.register('my_function.v1')
def my_function(x: np.ndarray) -> np.ndarray:
    "A custom function to normalise input data."
    return (x - x.mean()) / x.std()

# Load detector with config.toml file referencing "@my_function.v1"    
detector = load_detector(filepath)
```
````

````{grid-item-card}
:shadow: md
:margin: 1
:padding: 1
:columns: auto

**Specifying in a config.toml**
^^^

<p class="codeblock-label">config.toml</p>

```toml
name = "MMDDrift"
x_ref = "x_ref.npy"
preprocess_fn = "@my_function.v1"
```
````
`````

Once the custom function has been registered, it can be specified in `config.toml` files via its reference string
(with `@` prepended), for example `"@my_function.v1"` in this case. Other objects, such as custom tensorflow or 
pytorch models, can also be registered by using the `register` function directly. For example, to register a tensorflow 
encoder model:

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
from alibi_detect.utils.registry import registry

encoder_net = tf.keras.Sequential(
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

(examples)=
## Example config files

% To demonstrate the config-driven functionality, example detector configurations are presented in this section. 

% To download a config file and its related artefacts, click on the *Run Me* tabs, copy the Python code, and run it
% in your local Python shell.

(imdb_example)=
### Drift detection on text data

This example presents a configuration for the [MMDDrift](../api/alibi_detect.cd.mmd.rst) detector used in
[Text drift detection on IMDB movie reviews](../examples/cd_text_imdb.ipynb). The detector will pass the input text 
data through a `preprocess_fn` step consisting of a `tokenizer`, `embedding` and `model`. A
[Untrained AutoEncoder (UAE)](https://docs.seldon.io/projects/alibi-detect/en/latest/api/alibi_detect.cd.tensorflow.html?highlight=uae#alibi_detect.cd.tensorflow.UAE)
model is included in order to reduce the dimensionality of the embedding space, which consists of a 768-dimensional 
vector for each instance.


%````{tabbed} Config file
%:new-group:

<p class="codeblock-label">config.toml</p>

```toml
x_ref = "x_ref.npy"
name = "MMDDrift"

[preprocess_fn]
src = "@cd.tensorflow.preprocess.preprocess_drift"
batch_size = 32
max_len = 100
tokenizer.src = "tokenizer/"

[preprocess_fn.model]
src = "model/"

[preprocess_fn.embedding]
src = "embedding/"
type = "hidden_state"
layers = [-1, -2, -3, -4, -5, -6, -7, -8]
```
%````
%
%````{tabbed} Run Me
%
%```python
%from alibi_detect.utils.fetching import fetch_config
%from alibi_detect.utils.loading import load_detector
%filepath = 'IMDB_example_MMD/'
%fetch_config('imdb_mmd', filepath)
%detector = load_detector(filepath)
%```
%````

% TODO: Add a second example demo-ing loading of state (once implemented). e.g. for online or learned kernel.

## Advanced usage

### Validating config files

When `load_detector` is called, the `validate_config` utility function is used internally to validate the given
detector configuration. This allows any problems with the configuration to be detected prior to sometimes time-consuming
operations of loading artefacts and instantiating the detector. The `validate_config` function can also be used by 
devs working with Alibi Detect config dictionaries. 

Under-the-hood, `load_detector` parses the `config.toml` file into a *unresolved* config dictionary. It then passes
this *dict* through `validate_config`, to check for errors such as incorrectly named fields, and incorrect types. 
If working directly with config dictionaries, the same process can be done explicitly, for example:

```python
from alibi_detect.utils.loading import validate_config

# Define a simple config dict
cfg = {
    'name': 'MMDDrift',
    'x_ref': 'x_ref.npy',
    'p_val': [0.05],
    'bad_field': 'oops!'
}

# Validate the config
validate_config(cfg)
```

This will return a `ValidationError` because `p_val` is expected to be *float* not a *list*, and `bad_field` isn't 
a recognised field for the `MMDDrift` detector:

```console
ValidationError: 2 validation errors for MMDDriftConfig
p_val
  value is not a valid float (type=type_error.float)
bad_field
  extra fields not permitted (type=value_error.extra)
```

Validating at this stage is useful at errors can be caught before the sometimes time-consuming operation of 
resolving the config dictionary, which involves loading each artefact in the dictionary. The *resolved* config 
dictionary is then also passed through `validate_config`, and this second validation can also be done explicitly:

```python
import numpy as np
from alibi_detect.utils.loading import validate_config

# Create some reference data
x_ref = np.random.normal(size=(100,5))

# Define a simple config dict
cfg = {
    'name': 'MMDDrift',
    'x_ref': x_ref,
    'p_val': 0.05
}

# Validate the config
validate_config(cfg, resolved=True)
```

Note that since `resolved=True`, `validate_config` is now expecting `x_ref` to be a Numpy ndarray instead of a string.
This second level of validation can be useful as it helps detect problems with loaded artefacts before attempting the
sometimes time-consuming operation of instantiating the detector. 


%### Detector specification schemas
%
%Validation of detector config files is performed with [pydantic](https://pydantic-docs.helpmanual.io/). Each 
%detector's *unresolved* configuration is represented by a pydantic model, stored in `DETECTOR_CONFIGS`. Information on 
%a detector config's permitted fields and their types can be obtained via the config model's `schema` method
%(or `schema_json` if a json formatted string is preferred). For example, for the `KSDrift` detector:
%
%```python
%from alibi_detect.utils.schemas import DETECTOR_CONFIGS
%schema = DETECTOR_CONFIGS['KSDrift'].schema()
%
%```
%
%returns a dictionary with the keys `['title', 'type', 'properties', 'required', 'additionalProperties', 'definitions']`.
%The `'properties'` item is a dictionary containing all the possible fields for the detector:
%
%```python
%{'name': {'title': 'Name', 'type': 'string'},
% 'version': {'title': 'Version', 'default': '0.8.1dev', 'type': 'string'},
% 'config_spec': {'title': 'Config Spec',
%  'default': '0.1.0dev',
%  'type': 'string'},
% 'backend': {'title': 'Backend',
%  'default': 'tensorflow',
%  'enum': ['tensorflow', 'pytorch'],
%  'type': 'string'},
% 'x_ref': {'title': 'X Ref', 'default': 'x_ref.npy', 'type': 'string'},
% 'p_val': {'title': 'P Val', 'default': 0.05, 'type': 'number'},
% 'x_ref_preprocessed': {'title': 'X Ref Preprocessed',
%  'default': False,
%  'type': 'boolean'},
% ...
% }
%```
%
%Compulsory fields are listed in `'required'`, whilst additional pydantic model definitions such as 
%`ModelConfig` are stored in `'definitions'`. These additional models define the schemas for the 
%[artefact dictionaries](complex_fields). 
%
%Similarly, *resolved* detector configurations are represented by pydantic models stored in 
%`DETECTOR_CONFIGS_RESOLVED`. The difference being that for these models, artefacts are expected to have their 
%resolved types, for example `x_ref` should be a NumPy ndarray. The `schema` and `schema_json` methods can be applied 
%to these models in the same way.


```{note}
Sometimes, fields representing kwargs need to be set to `None`. However, unspecified fields are set to a detector's 
default kwargs (or for [Artefact dictionaries](dictionaries), the defaults shown in the tables). To set 
fields as `None`, specify them as the string `"None"`. 
```
