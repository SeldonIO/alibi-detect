from pytest_cases import param_fixture, parametrize, parametrize_with_cases

from alibi_detect.saving.tests.datasets import ContinuousData
from alibi_detect.saving.tests.models import encoder_model

from alibi_detect.cd.tensorflow import HiddenOutput as HiddenOutput_tf
from alibi_detect.saving.loading import _load_model_config, _load_optimizer_config
from alibi_detect.saving.saving import _path2str, _save_model_config
from alibi_detect.saving.schemas import ModelConfig

backend = param_fixture("backend", ['tensorflow'])


def test_load_optimizer_tf(backend):
    "Test the tensorflow _load_optimizer_config."
    class_name = 'Adam'
    learning_rate = 0.01
    epsilon = 1e-7
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
    assert type(optimizer).__name__ == class_name
    assert optimizer.learning_rate == learning_rate
    assert optimizer.epsilon == epsilon
    assert optimizer.amsgrad == amsgrad


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
