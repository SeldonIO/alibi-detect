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

For drift detectors, the `save_detector` serializes the detector into a human-readable specification file named `config.toml`, stored in `filepath`. As discussed further in [Config-driven detectors](config.md), config files can be edited before reloading the detector, and new detectors can be instantiated from hand-written config files. 

```{note}
At present outlier and adversarial detectors are serialized using the `dill` library, but these will also be be updated to use `config.toml` files in the future.
```

## Limitations
When loading a saved detector, a warning will be issued if the runtime alibi-detect version is 
different from the version used to save the detector. **It is highly recommended to use the same 
alibi-detect, Python and dependency versions as were used to save the detector to avoid potential 
bugs and incompatibilities**.
