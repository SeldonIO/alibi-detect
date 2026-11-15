"""
Keras 3 no longer accepts paths as the `fname` argument of tf.keras.utils.get_file. These
tests verify the fetching utils pass a bare filename as `fname` and the target directory
via `cache_dir` (with `cache_subdir=''` so files still land in the original directory).
"""
from pathlib import Path
from unittest.mock import patch

from alibi_detect.utils.fetching.fetching import _get_file, fetch_enc_dec, fetch_tf_model

GET_FILE = 'alibi_detect.utils.fetching.fetching.tf.keras.utils.get_file'
LOAD_MODEL = 'alibi_detect.utils.fetching.fetching.tf.keras.models.load_model'


def test_get_file_passes_basename_as_fname(tmp_path):
    filepath = tmp_path.joinpath('model', 'encoder_net.h5')
    with patch(GET_FILE) as get_file:
        _get_file(filepath, 'https://example.com/encoder_net.h5')
    kwargs = get_file.call_args.kwargs
    assert kwargs['fname'] == 'encoder_net.h5'
    assert Path(kwargs['cache_dir']).resolve() == filepath.parent.resolve()
    assert kwargs['cache_subdir'] == ''
    assert kwargs['origin'] == 'https://example.com/encoder_net.h5'


def test_fetch_tf_model_get_file_args():
    with patch(GET_FILE) as get_file, patch(LOAD_MODEL):
        fetch_tf_model('cifar10', 'resnet56')
    assert get_file.call_count == 1
    kwargs = get_file.call_args.kwargs
    assert kwargs['fname'] == Path(kwargs['fname']).name == 'resnet56.h5'
    assert Path(kwargs['cache_dir']).resolve() == Path.cwd()
    assert kwargs['cache_subdir'] == ''
    assert kwargs['origin'] == ('https://storage.googleapis.com/seldon-models/alibi-detect/'
                                'classifier/cifar10/resnet56/model.h5')


def test_fetch_enc_dec_get_file_args(tmp_path):
    with patch(GET_FILE) as get_file:
        fetch_enc_dec('https://example.com/detector', tmp_path)
    assert get_file.call_count == 2
    fnames = []
    for call in get_file.call_args_list:
        kwargs = call.kwargs
        assert kwargs['fname'] == Path(kwargs['fname']).name
        assert Path(kwargs['cache_dir']).resolve() == tmp_path.joinpath('model').resolve()
        assert kwargs['cache_subdir'] == ''
        fnames.append(kwargs['fname'])
    assert fnames == ['encoder_net.h5', 'decoder_net.h5']
