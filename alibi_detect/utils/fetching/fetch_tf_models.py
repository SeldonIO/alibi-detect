import logging
import os
from pathlib import Path
import dill
import tensorflow as tf
from tensorflow.python.keras import backend
from typing import Tuple, Union
from io import BytesIO
import requests
from requests import RequestException
from alibi_detect.models.tensorflow import PixelCNN
from alibi_detect.utils.fetching.url import _join_url

# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

logger = logging.getLogger(__name__)

"""Number of seconds to wait for URL requests before raising an error."""
TIMEOUT = 10


def get_pixelcnn_default_kwargs():
    dist = PixelCNN(
        image_shape=(28, 28, 1),
        num_resnet=5,
        num_hierarchies=2,
        num_filters=32,
        num_logistic_mix=1,
        receptive_field_dims=(3, 3),
        dropout_p=.3,
        l2_weight=0.
    )

    KWARGS_PIXELCNN = {
        'dist_s': dist,
        'dist_b': dist.copy(),
        'input_shape': (28, 28, 1)
    }
    return KWARGS_PIXELCNN


def fetch_tf_model(dataset: str, model: str) -> tf.keras.Model:
    """
    Fetch pretrained tensorflow models from the google cloud bucket.

    Parameters
    ----------
    dataset
        Dataset trained on.
    model
        Model name.

    Returns
    -------
    Pretrained tensorflow model.
    """
    url = 'https://storage.googleapis.com/seldon-models/alibi-detect/classifier/'
    path_model = _join_url(url, [dataset, model, 'model.h5'])
    save_path = tf.keras.utils.get_file(Path(model + '.h5').resolve(), path_model)
    if dataset == 'cifar10' and model == 'resnet56':
        custom_objects = {'backend': backend}
    else:
        custom_objects = None
    clf = tf.keras.models.load_model(save_path, custom_objects=custom_objects)
    return clf


def fetch_enc_dec(url: str, filepath: Union[str, os.PathLike]) -> None:
    """
    Download encoder and decoder networks.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    """
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    if not model_path.is_dir():
        model_path.mkdir(parents=True, exist_ok=True)
    # encoder and decoder
    tf.keras.utils.get_file(
        model_path.joinpath('encoder_net.h5'),
        _join_url(url_models, 'encoder_net.h5')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('decoder_net.h5'),
        _join_url(url_models, 'decoder_net.h5')
    )


def fetch_ae(url: str, filepath: Union[str, os.PathLike]) -> None:
    """
    Download AE outlier detector.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    """
    fetch_enc_dec(url, filepath)
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    # encoder and decoder
    tf.keras.utils.get_file(
        model_path.joinpath('checkpoint'),
        _join_url(url_models, 'checkpoint')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('ae.ckpt.index'),
        _join_url(url_models, 'ae.ckpt.index')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('ae.ckpt.data-00000-of-00001'),
        _join_url(url_models, 'ae.ckpt.data-00000-of-00001')
    )


def fetch_ad_ae(url: str, filepath: Union[str, os.PathLike], state_dict: dict) -> None:
    """
    Download AE adversarial detector.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    state_dict
        Dictionary containing the detector's parameters.
    """
    fetch_enc_dec(url, filepath)
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    tf.keras.utils.get_file(
        model_path.joinpath('model.h5'),
        _join_url(url_models, 'model.h5')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('checkpoint'),
        _join_url(url_models, 'checkpoint')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('ae.ckpt.index'),
        _join_url(url_models, 'ae.ckpt.index')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('ae.ckpt.data-00000-of-00002'),
        _join_url(url_models, 'ae.ckpt.data-00000-of-00002')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('ae.ckpt.data-00001-of-00002'),
        _join_url(url_models, 'ae.ckpt.data-00001-of-00002')
    )
    hidden_layer_kld = state_dict['hidden_layer_kld']
    if hidden_layer_kld:
        for i, (_, _) in enumerate(hidden_layer_kld.items()):
            hl = 'model_hl_' + str(i)
            tf.keras.utils.get_file(
                model_path.joinpath(hl + '.ckpt.index'),
                _join_url(url_models, hl + '.ckpt.index')
            )
            tf.keras.utils.get_file(
                model_path.joinpath(hl + '.ckpt.data-00000-of-00002'),
                _join_url(url_models, hl + '.ckpt.data-00000-of-00002')
            )
            tf.keras.utils.get_file(
                model_path.joinpath(hl + '.ckpt.data-00001-of-00002'),
                _join_url(url_models, hl + '.ckpt.data-00001-of-00002')
            )


def fetch_ad_md(url: str, filepath: Union[str, os.PathLike]) -> None:
    """
    Download model and distilled model.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    """
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    if not model_path.is_dir():
        model_path.mkdir(parents=True, exist_ok=True)
    # encoder and decoder
    tf.keras.utils.get_file(
        model_path.joinpath('model.h5'),
        _join_url(url_models, 'model.h5')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('distilled_model.h5'),
        _join_url(url_models, 'distilled_model.h5')
    )


def fetch_aegmm(url: str, filepath: Union[str, os.PathLike]) -> None:
    """
    Download AEGMM outlier detector.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    """
    # save encoder and decoder
    fetch_enc_dec(url, filepath)
    # save GMM network
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    tf.keras.utils.get_file(
        model_path.joinpath('gmm_density_net.h5'),
        _join_url(url_models, 'gmm_density_net.h5')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('checkpoint'),
        _join_url(url_models, 'checkpoint')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('aegmm.ckpt.index'),
        _join_url(url_models, 'aegmm.ckpt.index')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('aegmm.ckpt.data-00000-of-00001'),
        _join_url(url_models, 'aegmm.ckpt.data-00000-of-00001')
    )


def fetch_vae(url: str, filepath: Union[str, os.PathLike]) -> None:
    """
    Download VAE outlier detector.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    """
    fetch_enc_dec(url, filepath)
    # save VAE weights
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    tf.keras.utils.get_file(
        model_path.joinpath('checkpoint'),
        _join_url(url_models, 'checkpoint')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('vae.ckpt.index'),
        _join_url(url_models, 'vae.ckpt.index')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('vae.ckpt.data-00000-of-00001'),
        _join_url(url_models, 'vae.ckpt.data-00000-of-00001')
    )


def fetch_vaegmm(url: str, filepath: Union[str, os.PathLike]) -> None:
    """
    Download VAEGMM outlier detector.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    """
    # save encoder and decoder
    fetch_enc_dec(url, filepath)
    # save GMM network
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    tf.keras.utils.get_file(
        model_path.joinpath('gmm_density_net.h5'),
        _join_url(url_models, 'gmm_density_net.h5')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('checkpoint'),
        _join_url(url_models, 'checkpoint')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('vaegmm.ckpt.index'),
        _join_url(url_models, 'vaegmm.ckpt.index')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('vaegmm.ckpt.data-00000-of-00001'),
        _join_url(url_models, 'vaegmm.ckpt.data-00000-of-00001')
    )


def fetch_seq2seq(url: str, filepath: Union[str, os.PathLike]) -> None:
    """
    Download sequence-to-sequence outlier detector.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    """
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    if not model_path.is_dir():
        model_path.mkdir(parents=True, exist_ok=True)
    # save seq2seq
    tf.keras.utils.get_file(
        model_path.joinpath('checkpoint'),
        _join_url(url_models, 'checkpoint')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('seq2seq.ckpt.index'),
        _join_url(url_models, 'seq2seq.ckpt.index')
    )
    tf.keras.utils.get_file(
        model_path.joinpath('seq2seq.ckpt.data-00000-of-00001'),
        _join_url(url_models, 'seq2seq.ckpt.data-00000-of-00001')
    )
    # save threshold network
    tf.keras.utils.get_file(
        model_path.joinpath('threshold_net.h5'),
        _join_url(url_models, 'threshold_net.h5')
    )


def fetch_llr(url: str, filepath: Union[str, os.PathLike]) -> str:
    """
    Download Likelihood Ratio outlier detector.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    """
    url_models = _join_url(url, 'model')
    model_path = Path(filepath).joinpath('model').resolve()
    if not model_path.is_dir():
        model_path.mkdir(parents=True, exist_ok=True)
    try:
        tf.keras.utils.get_file(
            model_path.joinpath('model_s.h5'),
            _join_url(url_models, 'model_s.h5')
        )
        tf.keras.utils.get_file(
            model_path.joinpath('model_b.h5'),
            _join_url(url_models, 'model_b.h5')
        )
        model_type = 'weights'
        return model_type
    except Exception:
        tf.keras.utils.get_file(
            model_path.joinpath('model.h5'),
            _join_url(url_models, 'model.h5')
        )
        tf.keras.utils.get_file(
            model_path.joinpath('model_background.h5'),
            _join_url(url_models, 'model_background.h5')
        )
        return 'model'


def fetch_state_dict(url: str, filepath: Union[str, os.PathLike],
                     save_state_dict: bool = True) -> Tuple[dict, dict]:
    """
    Fetch the metadata and state/hyperparameter values of pre-trained detectors.

    Parameters
    ----------
    url
        URL to fetch detector from.
    filepath
        Local directory to save detector to.
    save_state_dict
        Whether to save the state dict locally.

    Returns
    -------
    Detector metadata and state.
    """
    # Check if metadata stored as dill or pickle
    try:
        url_meta = _join_url(url, 'meta.dill')
        resp = requests.get(url_meta, timeout=TIMEOUT)
        resp.raise_for_status()
        suffix = '.dill'
    except RequestException:
        try:
            url_meta = _join_url(url, 'meta.pickle')
            resp = requests.get(url_meta, timeout=TIMEOUT)
            resp.raise_for_status()
            suffix = '.pickle'
        except RequestException:
            logger.exception('Timed out while searching for meta.dill or meta.pickle files at %s.', url)
            raise

    # Load metadata and state_dict
    meta = dill.load(BytesIO(resp.content))
    try:
        url_state = _join_url(url, meta['name'] + suffix)
        resp = requests.get(url_state)
        resp.raise_for_status()
    except RequestException:
        logger.exception('Timed out while searching for corresponding state file at %s.', url)
        raise
    state_dict = dill.load(BytesIO(resp.content))

    # Save state
    if save_state_dict:
        filepath = Path(filepath)
        with open(filepath.joinpath('meta.dill'), 'wb') as f:
            dill.dump(meta, f)
        with open(filepath.joinpath(meta['name'] + '.dill'), 'wb') as f:
            dill.dump(state_dict, f)
    return meta, state_dict