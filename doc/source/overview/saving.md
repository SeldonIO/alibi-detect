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

To load a previously saved detector, use the `load_detector` method and provide it with the path to the detector's 
directory:

```python
from alibi_detect.utils.saving import load_detector

filepath = './my_detector/'
od = load_detector(filepath)
```

For drift detectors, `save_detector` serializes the detector via a config file named `config.toml`, stored in 
`filepath`. The [TOML](https://toml.io/en/) format is human-readable, which makes the config files useful for record 
keeping, and allows a detector to be edited before it is reloaded. For more information, see 
[Detector Configuration Files](config_files.md).

```{note}
At present outlier and adversarial detectors are serialized using the `dill` library, but these will also be be 
updated to use `config.toml` files in the future.
```

## Limitations

- When loading a saved detector, a warning will be issued if the runtime alibi-detect version is 
different from the version used to save the detector. **It is highly recommended to use the same 
alibi-detect, Python and dependency versions as were used to save the detector to avoid potential 
bugs and incompatibilities**.

- Save/load functionality is currently only implemented for offline drift detectors, using the tensorflow backend 
and/or tensorflow models. It will be updated to online detectors and pytorch detectors/models in the future.

