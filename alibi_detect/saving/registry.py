"""
This registry allows Python objects to be registered and accessed by their string reference later on. The primary usage
is to register objects so that they can be specified in a `config.toml` file. A number of Alibi Detect functions are
also pre-registered in the registry for convenience. See the
`Registering artefacts <https://docs.seldon.io/projects/alibi-detect/en/stable/overview/config_files.html#registering-artefacts>`_  # noqa: E501
documentation.

Examples
--------
Registering a simple function using the `@registry.register` decorator, and immediately fetching it:

.. code-block :: python

    import numpy as np
    from alibi_detect.saving import registry

    # Register a simple function
    @registry.register('my_function.v1')
    def my_function(x: np.ndarray) -> np.ndarray:
        "A custom function to normalise input data."
        return (x - x.mean()) / x.std()

    # Get function from registry
    fetched_function = registry.get('my_function.v1')

Instead of using a decorator, objects can also be registered by directly using the `registry.register()` function:

.. code-block :: python

    from alibi_detect.saving import registry

    my_object = ...
    registry.register("my_object.v1", func=my_object)
"""

import catalogue

from alibi_detect.utils.frameworks import has_pytorch, has_tensorflow, has_keops

if has_tensorflow:
    from alibi_detect.cd.tensorflow import \
        preprocess_drift as preprocess_drift_tf
    from alibi_detect.utils.tensorflow.data import TFDataset as TFDataset_tf
    from alibi_detect.utils.tensorflow.kernels import \
        GaussianRBF as GaussianRBF_tf, sigma_median as sigma_median_tf
    from alibi_detect.cd.tensorflow.context_aware import _sigma_median_diag as _sigma_median_diag_tf

if has_pytorch:
    from alibi_detect.cd.pytorch import \
        preprocess_drift as preprocess_drift_torch
    from alibi_detect.utils.pytorch.kernels import \
        GaussianRBF as GaussianRBF_torch, sigma_median as sigma_median_torch
    from alibi_detect.cd.pytorch.context_aware import _sigma_median_diag as _sigma_median_diag_torch

if has_keops:
    from alibi_detect.utils.keops.kernels import \
        GaussianRBF as GaussianRBF_keops, sigma_mean as sigma_mean_keops

# Create registry
registry = catalogue.create("alibi_detect", "registry")

# Register alibi-detect classes/functions
if has_tensorflow:
    registry.register('utils.tensorflow.kernels.GaussianRBF', func=GaussianRBF_tf)
    registry.register('utils.tensorflow.kernels.sigma_median', func=sigma_median_tf)
    registry.register('cd.tensorflow.context_aware._sigma_median_diag', func=_sigma_median_diag_tf)
    registry.register('cd.tensorflow.preprocess.preprocess_drift', func=preprocess_drift_tf)
    registry.register('utils.tensorflow.data.TFDataset', func=TFDataset_tf)

if has_pytorch:
    registry.register('utils.pytorch.kernels.GaussianRBF', func=GaussianRBF_torch)
    registry.register('utils.pytorch.kernels.sigma_median', func=sigma_median_torch)
    registry.register('cd.pytorch.context_aware._sigma_median_diag', func=_sigma_median_diag_torch)
    registry.register('cd.pytorch.preprocess.preprocess_drift', func=preprocess_drift_torch)

if has_keops:
    registry.register('utils.keops.kernels.GaussianRBF', func=GaussianRBF_keops)
    registry.register('utils.keops.kernels.sigma_mean', func=sigma_mean_keops)
