from pytest_cases import param_fixture, parametrize, parametrize_with_cases
import pytest

from alibi_detect.saving.tests.datasets import ContinuousData
from alibi_detect.saving.tests.models import encoder_model

from alibi_detect.cd.tensorflow import HiddenOutput as HiddenOutput_tf
from alibi_detect.saving.loading import _load_model_config, _load_optimizer_config
from alibi_detect.saving.saving import _path2str, _save_model_config, _save_optimizer_config
from alibi_detect.saving.schemas import ModelConfig, SupportedOptimizer
import tensorflow as tf
import numpy as np
from packaging import version

backend = param_fixture("backend", ['tensorflow'])


# Note: The full save/load functionality of optimizers (inc. validation) is tested in test_save_classifierdrift.
@pytest.mark.skipif(version.parse(tf.__version__) < version.parse('2.11.0'),
                    reason="Skipping since tensorflow < 2.11.0")
@parametrize('legacy', [True, False])
def test_load_optimizer_object_tf2pt11(legacy, backend):
    """
    Test the _load_optimizer_config with a tensorflow optimizer config. Only run if tensorflow>=2.11.

    Here we test that "new" and legacy optimizers can be saved/laoded. We expect the returned optimizer to be an
    instantiated `tf.keras.optimizers.Optimizer` object. Also test that the loaded optimizer can be saved.
    """
    class_name = 'Adam'
    class_str = class_name if legacy else 'Custom>' + class_name  # Note: see discussion in #739 re 'Custom>'
    learning_rate = np.float32(0.01)  # Set as float32 since this is what _save_optimizer_config returns
    epsilon = np.float32(1e-7)
    amsgrad = False

    # Load
    cfg_opt = {
        'class_name': class_str,
        'config': {
            'name': class_name,
            'learning_rate': learning_rate,
            'epsilon': epsilon,
            'amsgrad': amsgrad
        }
    }
    optimizer = _load_optimizer_config(cfg_opt, backend=backend)
    # Check optimizer
    SupportedOptimizer.validate_optimizer(optimizer, {'backend': 'tensorflow'})
    if legacy:
        assert isinstance(optimizer, tf.keras.optimizers.legacy.Optimizer)
    else:
        assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
    assert type(optimizer).__name__ == class_name
    assert optimizer.learning_rate == learning_rate
    assert optimizer.epsilon == epsilon
    assert optimizer.amsgrad == amsgrad

    # Save
    cfg_saved = _save_optimizer_config(optimizer)
    # Compare to original config
    for key, value in cfg_opt['config'].items():
        assert value == cfg_saved['config'][key]


@pytest.mark.skipif(version.parse(tf.__version__) >= version.parse('2.11.0'),
                    reason="Skipping since tensorflow >= 2.11.0")
def test_load_optimizer_object_tf_old(backend):
    """
    Test the _load_optimizer_config with a tensorflow optimizer config. Only run if tensorflow<2.11.

    We expect the returned optimizer to be an instantiated `tf.keras.optimizers.Optimizer` object.
    Also test that the loaded optimizer can be saved.
    """
    class_name = 'Adam'
    learning_rate = np.float32(0.01)  # Set as float32 since this is what _save_optimizer_config returns
    epsilon = np.float32(1e-7)
    amsgrad = False

    # Load
    cfg_opt = {
        'class_name': class_name,
        'config': {
            'name': class_name,
            'learning_rate': learning_rate,
            'epsilon': epsilon,
            'amsgrad': amsgrad
        }
    }
    optimizer = _load_optimizer_config(cfg_opt, backend=backend)
    # Check optimizer
    SupportedOptimizer.validate_optimizer(optimizer, {'backend': 'tensorflow'})
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
    assert type(optimizer).__name__ == class_name
    assert optimizer.learning_rate == learning_rate
    assert optimizer.epsilon == epsilon
    assert optimizer.amsgrad == amsgrad

    # Save
    cfg_saved = _save_optimizer_config(optimizer)
    # Compare to original config
    for key, value in cfg_opt['config'].items():
        assert value == cfg_saved['config'][key]


def test_load_optimizer_type(backend):
    """
    Test the _load_optimizer_config with just the `class_name` specified. In this case we expect a
    `tf.keras.optimizers.Optimizer` class to be returned.
    """
    class_name = 'Adam'
    cfg_opt = {'class_name': class_name}
    optimizer = _load_optimizer_config(cfg_opt, backend=backend)
    assert isinstance(optimizer, type)
    assert optimizer.__name__ == class_name


@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
@parametrize('model', [encoder_model])
@parametrize('layer', [None, -1])
def test_save_model_tf(data, model, layer, tmp_path):
    """
    Unit test for _save_model_config and _load_model_config with tensorflow model.
    """
    # Save model
    filepath = tmp_path
    input_shape = (data[0].shape[1],)
    cfg_model, _ = _save_model_config(model, base_path=filepath, input_shape=input_shape)
    cfg_model = _path2str(cfg_model)
    cfg_model = ModelConfig(**cfg_model).dict()
    assert tmp_path.joinpath('model').is_dir()
    assert tmp_path.joinpath('model/model.h5').is_file()

    # Adjust config
    cfg_model['src'] = tmp_path.joinpath('model')  # Need to manually set to absolute path here
    if layer is not None:
        cfg_model['layer'] = layer

    # Load model
    model_load = _load_model_config(cfg_model)
    if layer is None:
        assert isinstance(model_load, type(model))
    else:
        assert isinstance(model_load, HiddenOutput_tf)
