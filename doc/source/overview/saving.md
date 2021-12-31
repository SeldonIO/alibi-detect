# Saving and loading

Alibi Detect includes support for saving and loading detectors to disk. To 
save a detector, simply call the `save_detector` method and provide a path to a directory (a new
one will be created if it doesn't exist):

```python
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.saving import save_detector

od = OutlierVAE(...) 

filepath = './my_detector/'
save_detector(od, filepath)
```

To load a previously saved detector, use the `load_detector` method and provide it with the
path to the directory the detector is saved in:

```python
from alibi_detect.utils.saving import load_detector

filepath = './my_detector/'
od = load_detector(filepath)
```

## Detector config files

For drift detectors, `save_detector` serializes the detector via a config file named `config.toml`, stored in `filepath`. The human-readable [TOML](https://toml.io/en/) format means the config files can be useful for record keeping, and allows a detector to be edited before it is reloaded. 

```{note}
At present outlier and adversarial detectors are serialized using the `dill` library, but these will also be be updated to use `config.toml` files in the future.
```

New detectors can also be instantiated from *bare-bones* config files. For example, instead of the following code:


```python
import numpy as np
from alibi_detect.cd import MMDDrift

x_ref = np.load('x_ref.npy')
cd = MMDDrift(x_ref, p_val=0.05)
```

the same `MMDDrift` detector can be specified in a `config.toml` file as follows:

```toml
name = "MMDDrift"
x_ref = "x_ref.npy"
p_val = 0.05
```

where the fields following `name` are the args/kwargs to pass to the detector (see a detectors api docs for 
a full list of permissible args/kwargs e.g. [MMDDrift](../api/alibi_detect.cd.mmd.rst)). The `config.toml` file 
can then be loaded with `load_detector('config.toml)`. For consistency with the outlier and adversarial 
detector save/load functionality, a `filepath` can also be passed to `load_detector`, in which case it 
will search for a `config.toml` file within the specified directory. 


### Specifying artefacts

When specifying a detector via a `config.toml` file, the locally stored reference data `x_ref` must be specified. 
In addition, many detectors also require (or allow for) additional artefacts, such as kernels, functions and models. 
These can be specified in the `config.toml` file. For example, below is an example of the same detector
specification, with a `.dill` serialized function to preprocess reference and test data. 

```toml
name = "MMDDrift"
x_ref = "x_ref.npy"
p_val = 0.05
preprocess_fn = "funcion.dill"
```

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


- **Local files**: Artefacts may be specified as locally serialized files. e.g. `.dill` or `.npy`...

- **Function/object registry**: Discuss...

- **Nested dictionaries**: More complex artefacts are specified via dictionaries with `src` and additional option/setting fields (see below sections...). Discuss further here...

#### Preprocessing function

Simple `preprocess_fn`'s can be specified directly as a serialized dill file or via a function registry. If additional arguments are to be passed at call time, the `preprocess_fn`
can instead be specified via a nested config dictionary:

```toml
x_ref = "x_ref.npy"
p_val = 0.05

[preprocess_fn]
src = "function.dill"
kwargs = {kwarg1="a", kwarg2=1.5, kwarg3=true}
```

In this case, the `src` field can be a locally stored `.dill` file or a function registry, and `kwargs` is a dictionary of keyword arguments to be passed to the function.    
The [TOML](https://toml.io/en/) offers considerable flexibility in specifying nested fields, for example the `preprocess_fn.kwargs` field may also be specified as follows:

```toml
x_ref = "x_ref.npy"
p_val = 0.05

preprocess_fn.src = "function.dill"

[preprocess_fn.kwargs]
kwarg1 = "a"
kwarg2 = 1.5
kwarg3 = true
```

#### Models

Similar story for all below, elaborate...

#### Embedding

#### Tokenizers

#### Kernels

#### Optimizers

Dictionary in format used by `tf.keras.optimizers.serialize`

### Example config files

#### An LSDD drift detector with a text embedding and tokenizer
from the IMDB example

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

#### A classifier based drift detector

From the clf cifar10 example

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

### Advanced usage


#### Validating config files

Talk about pydantic. The unresolved and resolved config dict is validated with pydantic...

Public function for doing this...

#### Detector specification schemas

Can/should implement a public facing api to fetch the pydantic config schema's in json format. Discuss here...


## Limitations

- When loading a saved detector, a warning will be issued if the runtime alibi-detect version is 
different from the version used to save the detector. **It is highly recommended to use the same 
alibi-detect, Python and dependency versions as were used to save the detector to avoid potential 
bugs and incompatibilities**.

- Save/load functionality is currently only implemented for offline drift detectors, using the tensorflow backend and/or tensorflow models. It will be updated to online detectors and pytorch detectors/models in the future.

