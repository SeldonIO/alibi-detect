from pytest_cases import param_fixture, parametrize, parametrize_with_cases

from alibi_detect.saving.tests.datasets import ContinuousData
from alibi_detect.saving.tests.models import encoder_model

from alibi_detect.cd.pytorch import HiddenOutput as HiddenOutput_pt
from alibi_detect.saving.loading import _load_model_config, _load_optimizer_config
from alibi_detect.saving.saving import _path2str, _save_model_config
from alibi_detect.saving._pytorch.saving import save_device
from alibi_detect.saving.schemas import ModelConfig
import torch

backend = param_fixture("backend", ['pytorch'])


# Note: The full save/load functionality of optimizers (inc. validation) is tested in test_save_classifierdrift.
def test_load_optimizer(backend):
    """
    Test _load_optimizer_config with a pytorch optimizer, when the `torch.optim.Optimizer` class name is specified.
    For pytorch, we expect a `torch.optim` class to be returned.
    """
    class_name = 'Adam'
    cfg_opt = {'class_name': class_name}
    optimizer = _load_optimizer_config(cfg_opt, backend=backend)
    assert optimizer.__name__ == class_name
    assert isinstance(optimizer, type)


@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
@parametrize('model', [encoder_model])
@parametrize('layer', [None, -1])
def test_save_model_pt(data, model, layer, tmp_path):
    """
    Unit test for _save_model_config and _load_model_config with pytorch model.
    """
    # Save model
    filepath = tmp_path
    input_shape = (data[0].shape[1],)
    cfg_model, _ = _save_model_config(model, base_path=filepath, input_shape=input_shape)
    cfg_model = _path2str(cfg_model)
    cfg_model = ModelConfig(**cfg_model).dict()
    assert tmp_path.joinpath('model').is_dir()
    assert tmp_path.joinpath('model/model.pt').is_file()

    # Adjust config
    cfg_model['src'] = tmp_path.joinpath('model')  # Need to manually set to absolute path here
    if layer is not None:
        cfg_model['layer'] = layer

    # Load model
    model_load = _load_model_config(cfg_model)
    if layer is None:
        assert isinstance(model_load, type(model))
    else:
        assert isinstance(model_load, HiddenOutput_pt)


@parametrize('device', ['cpu', 'gpu', 'cuda', 'cuda:0', torch.device('cuda'), torch.device('cuda:0')])
def test_save_device_pt(device):
    """
    Unit test for _save_device.
    """
    result = save_device(device)
    assert result in {'gpu', 'cuda', 'cpu'}
