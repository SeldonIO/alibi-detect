from pathlib import Path
from unittest.mock import MagicMock, patch

from alibi_detect.utils.fetching import fetch_tf_model


@patch("alibi_detect.utils.fetching.fetching.tf.keras.models.load_model")
@patch("alibi_detect.utils.fetching.fetching.tf.keras.utils.get_file")
def test_fetch_tf_model_uses_filename_and_cache_dir(
    mock_get_file: MagicMock,
    mock_load_model: MagicMock,
) -> None:
    mock_get_file.return_value = "/tmp/resnet32.h5"

    fetch_tf_model("cifar10", "resnet32")

    mock_get_file.assert_called_once()

    _, kwargs = mock_get_file.call_args

    assert kwargs["fname"] == "resnet32.h5"
    assert kwargs["cache_dir"] == Path.cwd()
    assert kwargs["cache_subdir"] == ""
    assert "origin" in kwargs

    mock_load_model.assert_called_once()

    args, kwargs = mock_load_model.call_args

    assert args[0] == "/tmp/resnet32.h5"
    assert kwargs["custom_objects"] is None
