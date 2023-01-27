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

# Error/warning messages to check for (just snippets are enough to check...)
INPUT_SHAPE_MSG = "model's input shape is not available."
CUSTOM_OBJ_MSG = "The TensorFlow model may have been loaded incorrectly."


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
@parametrize('model', [encoder_model])
@parametrize('layer', [None, -1])
def test_save_model_tf(data, model, layer, tmp_path):
    """
    Unit test for _save_model_config and _load_model_config with tensorflow model.
    """
    # Save model
    filepath = tmp_path
    cfg_model, _ = _save_model_config(model, base_path=filepath)
    cfg_model = _path2str(cfg_model)
    cfg_model = ModelConfig(**cfg_model).dict()
    assert tmp_path.joinpath('model').is_dir()

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


@parametrize_with_cases("data", cases=ContinuousData.data_synthetic_nd, prefix='data_')
@parametrize("call, pass_custom_objects", [
    (True, True),  # call model before saving and pass custom_objects - expected to pass
    (False, True),  # Don't call model before saving - expected to raise error
    (True, False)  # Don't pass custom objects, expected to raise warning (prior to tensorflow raising error)
    ]
)
@parametrize('model', [encoder_model_subclassed])
def test_save_model_tf_subclassed(data, call, pass_custom_objects, model, tmp_path):
    """
    Unit test for _save_model_config and _load_model_config with a subclassed tensorflow model.

    For a subclassed model to be saved/loaded it must 1) have been called/built 2) have valid `get_config` and
    `from_config` methods 3) have any custom objects provided at load time. If these conditions are not met alibi-detect
    raises various errors and warnings (see the TensorFlow tab in
    https://docs.seldon.io/projects/alibi-detect/en/stable/overview/saving.html#supported-ml-models).

    This test checks three different scenarios:

        1. Model called prior to saving, and `custom_objects` correctly passed. This should pass with no
        warnings/errors.
        2. Model not called before saving. `alibi_detect.saving._tensorflow.save_model` should raise `ValueError`.
        3. `custom_objects` not passed to `_load_model_config`. `alibi_detect.saving._tensorflow.load_model` should
        raise `UserWarning`.
    """
    # Save model
    filepath = tmp_path
    if call:
        # Call model, then save
        X, _ = data
        model(X)
        cfg_model, _ = _save_model_config(model, base_path=filepath)
    else:
        # Don't call model, check correct error raised when saving
        with pytest.raises(ValueError, match=INPUT_SHAPE_MSG):
            _save_model_config(model, base_path=filepath)
        return  # Skip the rest of the test

    # Parse and check config
    cfg_model = _path2str(cfg_model)
    cfg_model = ModelConfig(**cfg_model).dict()
    assert tmp_path.joinpath('model').is_dir()

    # Adjust config
    cfg_model['src'] = tmp_path.joinpath('model')  # Need to manually set to absolute path here

    # Load model
    if pass_custom_objects:
        # Load model with custom objects passed
        model_load = _load_model_config(cfg_model, custom_objects={'EncoderTF': EncoderTF})
        # If there was a problem w/ passing custom_objects, model will be a `keras.saving.saved_model.load.EncoderTF`
        # object instead of the original `EncoderTF` object
        assert isinstance(model_load, type(model))
    else:
        # Don't load model, check correct warning raised
        with pytest.warns(UserWarning, match=CUSTOM_OBJ_MSG):
            _load_model_config(cfg_model)
        return  # Skip the rest of the test


@parametrize_with_cases("data", cases=ContinuousData, prefix='data_')
@parametrize("run_predict, pass_custom_objects", [
    (True, True),  # Run predict before saving and pass custom_objects - expected to pass
    (False, True),  # Don't run predict before saving - expected to raise error
    (True, False)  # Don't pass custom objects, expected to raise warning (prior to tensorflow raising error)
    ]
)
@parametrize("model", [classifier_model_subclassed])
def test_save_classifierdrift_subclassed(data, model, run_predict, pass_custom_objects, tmp_path):  # noqa: F811
    """
    Copy of `test_save_classifierdrift` to specifically test use with a subclassed tensorflow model, with the custom
    model class given to `load_detector`.

    The same three scenarios tested in `test_save_model_tf_subclassed` are tested here, but with the detector's
    `predict` method called instead of the model directly. This is repeated here to check that the correct model
    related errors/warnings are still raised prior to the final `RuntimeError` being raised when `save_detector` fails.
    """
    # Init detector and predict
    X_ref, X_h0 = data
    cd = ClassifierDrift(X_ref,
                         model=model,
                         backend='tensorflow')
    if run_predict:
        preds = cd.predict(X_h0)  # noqa: F841
        save_detector(cd, tmp_path)
    else:
        # We expect a RunTimeError (when any errors occur during saving clean up is performed and then this is raised)
        with pytest.raises(RuntimeError) as excinfo:
            save_detector(cd, tmp_path)
        # Check that the underlying error is the expected ValueError (and check a snippet of the error message)
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert INPUT_SHAPE_MSG in str(excinfo.value.__cause__)
        return  # Skip the rest of the test

    # Load detector and make another prediction
    if pass_custom_objects:
        cd_load = load_detector(tmp_path, custom_objects={'ClassifierTF': ClassifierTF})
    else:
        # Check the expected UserWarning is raised by searching for a snippet of the warning message
        with pytest.warns(UserWarning, match=CUSTOM_OBJ_MSG):
            with pytest.raises(Exception):  # Need to catch this as tf raises an error after the warning in this case
                load_detector(tmp_path)
        return  # Skip the rest of the test

    preds_load = cd_load.predict(X_h0)  # noqa: F841

    # Assert
    np.testing.assert_array_equal(X_ref, cd_load._detector.x_ref)
    assert isinstance(cd_load._detector.model, ClassifierTF)
