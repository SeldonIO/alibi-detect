import pytest
from pytest_cases import param_fixture, parametrize, parametrize_with_cases

from alibi_detect.cd import ClassifierDrift
from alibi_detect.saving.tests.datasets import ContinuousData
from alibi_detect.saving.tests.models import encoder_model, encoder_model_subclassed, EncoderTF,\
    classifier_model_subclassed, ClassifierTF

from alibi_detect.cd.tensorflow import HiddenOutput as HiddenOutput_tf
from alibi_detect.saving import save_detector, load_detector
from alibi_detect.saving.loading import _load_model_config, _load_optimizer_config
from alibi_detect.saving.saving import _path2str, _save_model_config
from alibi_detect.saving.schemas import ModelConfig
import tensorflow as tf
import numpy as np

backend = param_fixture("backend", ['tensorflow'])


# Note: The full save/load functionality of optimizers (inc. validation) is tested in test_save_classifierdrift.
def test_load_optimizer_object(backend):
    """
    Test the _load_optimizer_config with a tensorflow optimizer config. In this case, we expect the returned optimizer
    to be an instantiated `tf.keras.optimizers.Optimizer` object.
    """
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
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
    assert type(optimizer).__name__ == class_name
    assert optimizer.learning_rate == learning_rate
    assert optimizer.epsilon == epsilon
    assert optimizer.amsgrad == amsgrad


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
@parametrize('model', [encoder_model, encoder_model_subclassed])
@parametrize('layer', [None, -1])
def test_save_model_tf(data, model, layer, tmp_path):
    """
    Unit test for _save_model_config and _load_model_config with tensorflow model.
    """
    if layer is not None and isinstance(model, EncoderTF):
        pytest.skip("Don't test `layer != None` when the model is a subclassed model.")

    # Save model
    filepath = tmp_path
    input_shape = (data[0].shape[1],)
    cfg_model, _ = _save_model_config(model, base_path=filepath, input_shape=input_shape)
    cfg_model = _path2str(cfg_model)
    cfg_model = ModelConfig(**cfg_model).dict()
    assert tmp_path.joinpath('model').is_dir()

    # Adjust config
    cfg_model['src'] = tmp_path.joinpath('model')  # Need to manually set to absolute path here
    if layer is not None:
        cfg_model['layer'] = layer

    # Load model
    kwargs = {}
    if isinstance(model, EncoderTF):
        kwargs['custom_objects'] = {'EncoderTF': EncoderTF}
    model_load = _load_model_config(cfg_model, **kwargs)
    if layer is None:
        # Note: If something went wrong with passing `custom_objects` to the `tf.keras.models.load_model`
        #  (in `_load_model_config`). The model will likely be a `keras.saving.saved_model.load.EncoderTF` object
        #  instead of the `EncoderTF` object defined in `alibi_detect.saving.tests.models` (and below will fail)
        assert isinstance(model_load, type(model))
    else:
        assert isinstance(model_load, HiddenOutput_tf)


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
@parametrize("model", [classifier_model_subclassed])
def test_save_classifierdrift_subclassed(data, model, tmp_path):  # noqa: F811
    """
    Copy of `test_save_classifierdrift` to specifically test use with a subclassed tensorflow model, with the custom
    model class given to `load_detector`.
    """
    # Init detector and predict

    X_ref, X_h0 = data
    cd = ClassifierDrift(X_ref,
                         model=model,
                         backend='tensorflow')
    preds = cd.predict(X_h0)  # noqa: F841
    save_detector(cd, tmp_path)

    # Load detector and make another prediction
    custom_objects = {'ClassifierTF': ClassifierTF}
    cd_load = load_detector(tmp_path, custom_objects=custom_objects)
    preds_load = cd_load.predict(X_h0)  # noqa: F841

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load._detector.x_ref)
    assert isinstance(cd_load._detector.model, ClassifierTF)
