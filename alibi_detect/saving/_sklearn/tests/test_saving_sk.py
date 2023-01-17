from pytest_cases import param_fixture, parametrize, parametrize_with_cases

from alibi_detect.saving.tests.datasets import ContinuousData
from alibi_detect.saving.tests.models import classifier_model, xgb_classifier_model

from alibi_detect.saving.loading import _load_model_config
from alibi_detect.saving.saving import _path2str, _save_model_config
from alibi_detect.saving.schemas import ModelConfig

backend = param_fixture("backend", ['sklearn'])


@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
@parametrize('model', [classifier_model, xgb_classifier_model])
def test_save_model_sk(data, model, tmp_path):
    """
    Unit test for _save_model_config and _load_model_config with scikit-learn and xgboost model.
    """
    # Save model
    filepath = tmp_path
    cfg_model, _ = _save_model_config(model, base_path=filepath)
    cfg_model = _path2str(cfg_model)
    cfg_model = ModelConfig(**cfg_model).dict()
    assert tmp_path.joinpath('model').is_dir()
    assert tmp_path.joinpath('model/model.joblib').is_file()

    # Adjust config
    cfg_model['src'] = tmp_path.joinpath('model')  # Need to manually set to absolute path here

    # Load model
    model_load = _load_model_config(cfg_model)
    assert isinstance(model_load, type(model))
