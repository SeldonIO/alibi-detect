# Config-driven detectors

Instead of instantiating Alibi Detect detectors at runtime in the usual way... blah blah blah

```python
from alibi_detect.cd import MMDDrift

x_ref = ...
cd = MMDDrift(x_ref, p_val=0.05)
```

Detectors can also be instantiated via a configuration file named `config.toml`. Follows usual toml spec

```toml
x_ref = "x_ref.npy"
p_val = 0.05
```

```python
cd = load_detector('config.toml')
```

## Configuring complex artefacts

```toml
x_ref = "x_ref.npy"
p_val = 0.05
preprocess_fn = ...
```

```toml
x_ref = "x_ref.npy"
p_val = 0.05

[preprocess_fn]
src = ...
kwargs = 
```

### Local files

### Function/object registry

### Passing kwargs 

## Advanced usage


### Validating config files

### Getting pydantic model schemas


```{note}
This capability is currently only implemented for offline drift detectors, using the tensorflow backend and/or tensorflow models. It will be updated to online detectors and pytorch detectors/models in the future.
```
