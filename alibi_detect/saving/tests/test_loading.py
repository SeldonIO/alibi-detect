import numpy as np
import pytest

from alibi_detect.cd import TabularDrift
from alibi_detect.saving import save_detector, load_detector


def test_loading_detector_with_plain_data(tmp_path):
    data = np.array([[42, 1, 1234077000], [42, 2, 1234088000], [42, 3, 1234099000]])
    p_val = 0.42

    create_and_save_detector(data, p_val, tmp_path)

    loaded_detector = load_detector(tmp_path)

    assert loaded_detector.get_config()["name"] == "TabularDrift"
    assert loaded_detector.get_config()["p_val"] == p_val
    np.testing.assert_array_equal(loaded_detector.get_config()["x_ref"], data)


def test_loading_detector_with_data_containing_objects_throws_exception(tmp_path):
    data = np.array([['42', 1, 1234077000], ['42', 2, 1234088000], ['42', 3, 1234099000]], dtype=object)
    p_val = 0.42

    create_and_save_detector(data, p_val, tmp_path)

    with pytest.raises(Exception) as ex_info:
        _ = load_detector(tmp_path)

    assert ex_info.typename == "ValueError"
    assert ex_info.value.args[0] == "Object arrays cannot be loaded when allow_pickle=False"


def test_loading_detector_with_data_containing_objects(tmp_path):
    data = np.array([['42', 1, 1234077000], ['42', 2, 1234088000], ['42', 3, 1234099000]], dtype=object)
    p_val = 0.42

    create_and_save_detector(data, p_val, tmp_path)

    loaded_detector = load_detector(tmp_path, enable_unsafe_loading=True)

    assert loaded_detector.get_config()["name"] == "TabularDrift"
    assert loaded_detector.get_config()["p_val"] == p_val
    np.testing.assert_array_equal(loaded_detector.get_config()["x_ref"], data)


def create_and_save_detector(data: np.ndarray, p_val: float, path):
    detector = TabularDrift(
        x_ref=data,
        p_val=p_val,
        x_ref_preprocessed=True
    )

    save_detector(
        detector,
        path
    )
