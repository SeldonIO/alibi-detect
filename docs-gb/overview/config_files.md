# Detector Configuration Files

For advanced use cases, Alibi Detect features powerful configuration file based functionality. As shown below,
**Drift detectors** can be specified with a configuration file named `config.toml` (adversarial and outlier 
detectors coming soon!), which can then be passed to {func}`~alibi_detect.saving.load_detector`:


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
from alibi_detect.saving import load_detector
filepath = 'detector_directory/'
detector = load_detector(filepath)
```
````
`````

Compared to *standard instantiation*, config-driven instantiation has a number of advantages:

- **Human readable**: The `config.toml` files are human-readable (and editable!), providing a readily accessible record 
of previously created detectors.
- **Flexible artefact specification**: Artefacts such as datasets and models can be specified as locally serialized
objects, or as runtime registered objects (see [Specifying complex fields](complex_fields)). Multiple detectors can 
share the same artefacts, and they can be easily swapped.
- **Inbuilt validation**: The {func}`~alibi_detect.saving.load_detector` function uses
[pydantic](https://pydantic-docs.helpmanual.io/) to validate detector configurations.

To get a general idea of the expected layout of a config file, see the [Example config files](examples). Alternatively, 
to obtain a fully populated config file for reference, users can run one of the [example notebooks](../cd/examples.md) 
and generate a config file by passing an instantiated detector to {func}`~alibi_detect.saving.save_detector`.

## Configuration file layout

All detector configuration files follow a consistent layout, simplifying the process of writing simple config files
by hand. For example, a {class}`~alibi_detect.cd.KSDrift` detector with a 
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
from alibi_detect.saving import load_detector
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
The remaining fields are the args/kwargs to pass to the detector (see the {mod}`alibi_detect.cd` docs
for a full list of permissible args/kwargs for each detector). All config fields
follow this convention, however as discussed in [Specifying artefacts](complex_fields), some fields can be 
more complex than others. 


```{note}
In the  above example `config.toml`, `x_ref` and `preprocess_fn` are stored in `detector_directory/`, but this directory
isn't included in the config file. This is because in the config file, **relative directories are relative to the 
location of the config.toml file**. Filepaths may be absolute, or include nested directories, but **must be POSIX 
paths** i.e. use `/` path separators instead of `\`.
```

```{note}
Sometimes, fields representing kwargs need to be set to `None`. However, unspecified fields are set to a detector's 
default kwargs (or for [Artefact dictionaries](dictionaries), the defaults shown in the tables). To set 
fields as `None`, specify them as the string `"None"`. 
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
objects defined at runtime can be registered using {func}`alibi_detect.saving.registry`, allowing them to be specified 
in the config file without having to serialise them. For convenience a number of Alibi Detect functions such as 
{func}`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift` are also pre-registered. 

- **Dictionaries**: More complex artefacts are specified via nested dictionaries, usually containing a `src` field and 
additional option/setting fields. Sometimes these fields may be nested artefact dictionaries themselves. See 
[Artefact dictionaries](dictionaries) for further details.

The following table shows the allowable formats for all possible config file artefacts.

```{table} Allowable artefact formats
:name: all-artefacts-table

|Field                     |.npy file  |.dill file  |[Registry](registering_artefacts)|[Artefact Dictionary](dictionaries)                                                                         | 
|:-------------------------|:---------:|:----------:|:-------------------------------:|:----------------------------------------------------------------------------------------------------------:|
|`x_ref`                   |✔          |            |                                 |                                                                                                            |
|`c_ref`                   |✔          |            |                                 |                                                                                                            |
|`reg_loss_fn`             |           |✔           |✔                                |                                                                                                            |
|`dataset`                 |           |✔           |✔                                |                                                                                                            |
|`initial_diffs`           |✔          |            |                                 |                                                                                                            |
|`model`/`proj`            |           |            |✔                                |{class}`~alibi_detect.saving.schemas.ModelConfig`                                                           |
|`preprocess_fn`           |           |✔           |✔                                |{class}`~alibi_detect.saving.schemas.PreprocessConfig`                                                      |
|`preprocess_batch_fn`     |           |✔           |✔                                |                                                                                                            |
|`embedding`               |           |            |✔                                |{class}`~alibi_detect.saving.schemas.EmbeddingConfig`                                                       |
|`tokenizer`               |           |            |✔                                |{class}`~alibi_detect.saving.schemas.TokenizerConfig`                                                       |
|`kernel`                  |           |            |✔                                |{class}`~alibi_detect.saving.schemas.KernelConfig` or {class}`~alibi_detect.saving.schemas.DeepKernelConfig`|
|`kernel_a`/`kernel_b`     |           |            |✔                                |{class}`~alibi_detect.saving.schemas.KernelConfig`                                                          |
|`optimizer`               |           |✔           |✔                                | {class}`~alibi_detect.saving.schemas.OptimizerConfig`                                                      |
```

(dictionaries)=
### Artefact dictionaries

Simple artefacts, for example a simple preprocessing function serialized in a dill file, can be specified directly:
`preprocess_fn = "function.dill"`. However, if more complex, they can be specified as an *artefact dictionary*:

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[preprocess_fn]
src = "function.dill"
kwargs = {'kwarg1'=42, 'kwarg2'=false}
```

Here, the `preprocess_fn` field is a {class}`~alibi_detect.saving.schemas.PreprocessConfig` artefact dictionary.
In this example, specifying the `preprocess_fn` function as a dictionary allows us to specify additional `kwarg`'s to 
be passed to the function upon loading. This example also demonstrates the flexibility of the TOML format, with
dictionaries able to be specified with {} brackets or by sections demarcated with [] brackets (see the 
[TOML documentation](https://toml.io/en/) for more details on the TOML format).

Other config fields in the {ref}`all-artefacts-table` table can be specified via artefact dictionaries in a similar way. 
For example, the `model` and `proj` fields can be set as TensorFlow or PyTorch
models via the {class}`~alibi_detect.saving.schemas.ModelConfig` dictionary. Often an artefact dictionary may itself
contain nested artefact dictionaries, as is the case in in the following example, where a `preprocess_fn` is specified 
with a TensorFlow `model`.

<p class="codeblock-label">config.toml (excerpt)</p>

```toml
[preprocess_fn]
src = "@cd.tensorflow.preprocess.preprocess_drift"
batch_size = 32

[preprocess_fn.model]
src = "model/"
```

Each artefact dictionary has an associated pydantic model which is used for [validation of config files](validation). 
The [documentation](../api/alibi_detect.saving.schemas.rst) for these pydantic models provides a description of the 
permissible fields for each artefact dictionary. For examples of how the artefact dictionaries can be used in practice, 
see {ref}`examples`.


(registering_artefacts)=
### Registering artefacts

Custom artefacts defined in Python code may be specified in the config file without the need to serialise them, 
by first adding them to the Alibi Detect artefact registry using the {mod}`alibi_detect.saving.registry` 
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
from alibi_detect.saving import registry, load_detector

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
from alibi_detect.saving import registry

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

| Function/Class                                                        | Registry reference*                           | Tensorflow  | Pytorch  | 
|:----------------------------------------------------------------------|:----------------------------------------------|:-----------:|:--------:|
| {func}`~alibi_detect.cd.tensorflow.preprocess.preprocess_drift`       | `'@cd.[backend].preprocess.preprocess_drift'` |      ✔      |    ✔     |
| {class}`~alibi_detect.utils.tensorflow.kernels.GaussianRBF`           | `'@utils.[backend].kernels.GaussianRBF'`      |      ✔      |    ✔     |
| {class}`~alibi_detect.utils.tensorflow.data.TFDataset`                | `'@utils.tensorflow.data.TFDataset'`          |      ✔      |          |

**For backend-specific functions/classes, [backend] should be replaced the desired backend e.g. `tensorflow` or `pytorch`.*

These can be used in `config.toml` files. Of particular importance are the `preprocess_drift` utility functions, 
which allows models, tokenizers and embeddings to be easily specified for preprocessing, as demonstrated in the 
[IMDB example](imdb_example).

(examples)=
## Example config files

% To demonstrate the config-driven functionality, example detector configurations are presented in this section. 

% To download a config file and its related artefacts, click on the *Run Me* tabs, copy the Python code, and run it
% in your local Python shell.

(imdb_example)=
### Drift detection on text data

This example presents a configuration for the {class}`~alibi_detect.cd.MMDDrift` detector used in
[Text drift detection on IMDB movie reviews](../examples/cd_text_imdb.ipynb). The detector will pass the input text 
data through a `preprocess_fn` step consisting of a `tokenizer`, `embedding` and `model`. An
[Untrained AutoEncoder (UAE)](https://docs.seldon.io/projects/alibi-detect/en/stable/api/alibi_detect.cd.tensorflow.html?highlight=uae#alibi_detect.cd.tensorflow.UAE)
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
%from alibi_detect.saving import load_detector
%filepath = 'IMDB_example_MMD/'
%fetch_config('imdb_mmd', filepath)
%detector = load_detector(filepath)
%```
%````

% TODO: Add a second example demo-ing loading of state (once implemented). e.g. for online or learned kernel.

%## Advanced usage

(validation)=
## Validating config files

When {func}`~alibi_detect.saving.load_detector` is called, the {func}`~alibi_detect.saving.validate_config`
utility function is used internally to validate the given detector configuration. This allows any problems with the 
configuration to be detected prior to sometimes time-consuming operations of loading artefacts and instantiating the 
detector. {func}`~alibi_detect.saving.validate_config` can also be used by devs working with Alibi Detect config 
dictionaries. 

Under-the-hood, {func}`~alibi_detect.saving.load_detector` parses the `config.toml` file into a *unresolved* 
config dictionary. It then passes this *dict* through {func}`~alibi_detect.saving.validate_config`, to check for 
errors such as incorrectly named fields, and incorrect types. If working directly with config dictionaries, the same 
process can be done explicitly, for example:

```python
from alibi_detect.saving import validate_config

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

Validating at this stage is useful as errors can be caught before the sometimes time-consuming operation of 
resolving the config dictionary, which involves loading each artefact in the dictionary 
({func}`~alibi_detect.saving.read_config` and {func}`~alibi_detect.saving.resolve_config` can be used to manually read 
and resolve a config for debugging). The *resolved* config dictionary is then also passed through 
{func}`~alibi_detect.saving.validate_config`, and this second validation can also be done explicitly:

```python
import numpy as np
from alibi_detect.saving import validate_config

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

Note that since `resolved=True`, {func}`~alibi_detect.saving.validate_config` is now expecting `x_ref` to be a 
Numpy ndarray instead of a string. This second level of validation can be useful as it helps detect problems with loaded 
artefacts before attempting the sometimes time-consuming operation of instantiating the detector. 

%### Detector specification schemas
%
%Validation of detector config files is performed with [pydantic](https://pydantic-docs.helpmanual.io/). Each 
%detector's *unresolved* configuration is represented by a pydantic model, stored in `DETECTOR_CONFIGS`. Information on 
%a detector config's permitted fields and their types can be obtained via the config model's `schema` method
%(or `schema_json` if a json formatted string is preferred). For example, for the `KSDrift` detector:
%
%```python
%from alibi_detect.saving.schemas import DETECTOR_CONFIGS
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
