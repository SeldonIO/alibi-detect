import os
from pathlib import Path
from typing import Union, TYPE_CHECKING

from tensorflow.python.keras import backend

if TYPE_CHECKING:
    from alibi_detect.ad.adversarialae import AdversarialAE  # noqa
    from alibi_detect.ad.model_distillation import ModelDistillation  # noqa
    from alibi_detect.base import BaseDetector  # noqa
    from alibi_detect.od.llr import LLR  # noqa
    from alibi_detect.od.isolationforest import IForest  # noqa
    from alibi_detect.od.mahalanobis import Mahalanobis  # noqa
    from alibi_detect.od.aegmm import OutlierAEGMM  # noqa
    from alibi_detect.od.ae import OutlierAE  # noqa
    from alibi_detect.od.prophet import OutlierProphet  # noqa
    from alibi_detect.od.seq2seq import OutlierSeq2Seq  # noqa
    from alibi_detect.od.vae import OutlierVAE  # noqa
    from alibi_detect.od.vaegmm import OutlierVAEGMM  # noqa
    from alibi_detect.od.sr import SpectralResidual  # noqa

from alibi_detect.utils.fetching.fetch_tf_models import logger, fetch_state_dict, fetch_ae, fetch_aegmm, \
    fetch_vae, fetch_vaegmm, fetch_seq2seq, fetch_ad_ae, fetch_ad_md, fetch_llr, get_pixelcnn_default_kwargs
from alibi_detect.utils.fetching.url import _join_url
from alibi_detect.utils.saving import load_detector

Data = Union[
    'BaseDetector',
    'AdversarialAE',
    'ModelDistillation',
    'IForest',
    'LLR',
    'Mahalanobis',
    'OutlierAEGMM',
    'OutlierAE',
    'OutlierProphet',
    'OutlierSeq2Seq',
    'OutlierVAE',
    'OutlierVAEGMM',
    'SpectralResidual'
]


def fetch_detector(filepath: Union[str, os.PathLike],
                   detector_type: str,
                   dataset: str,
                   detector_name: str,
                   model: str = None) -> Data:
    """
    Fetch an outlier or adversarial detector from a google bucket, save it locally and return
    the initialised detector.

    Parameters
    ----------
    filepath
        Local directory to save detector to.
    detector_type
        `outlier` or `adversarial`.
    dataset
        Dataset of pre-trained detector. E.g. `kddcup`, `cifar10` or `ecg`.
    detector_name
        Name of the detector in the bucket.
    model
        Classification model used for adversarial detection.

    Returns
    -------
    Initialised pre-trained detector.
    """
    # create path (if needed)
    filepath = Path(filepath)
    if not filepath.is_dir():
        filepath.mkdir(parents=True, exist_ok=True)
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
    # create url of detector
    url = 'https://storage.googleapis.com/seldon-models/alibi-detect/'
    if detector_type == 'adversarial':
        url = _join_url(url, ['ad', dataset, model, detector_name])
    elif detector_type == 'outlier':
        url = _join_url(url, ['od', detector_name, dataset])

    # fetch the metadata and state dict
    meta, state_dict = fetch_state_dict(url, filepath, save_state_dict=True)

    # load detector
    name = meta['name']
    kwargs = {}  # type: dict
    if name == 'OutlierAE':
        fetch_ae(url, filepath)
    elif name == 'OutlierAEGMM':
        fetch_aegmm(url, filepath)
    elif name == 'OutlierVAE':
        fetch_vae(url, filepath)
    elif name == 'OutlierVAEGMM':
        fetch_vaegmm(url, filepath)
    elif name == 'OutlierSeq2Seq':
        fetch_seq2seq(url, filepath)
    elif name == 'AdversarialAE':
        fetch_ad_ae(url, filepath, state_dict)
        if model == 'resnet56':
            kwargs = {'custom_objects': {'backend': backend}}
    elif name == 'ModelDistillation':
        fetch_ad_md(url, filepath)
        if model == 'resnet56':
            kwargs = {'custom_objects': {'backend': backend}}
    elif name == 'LLR':
        model_type = fetch_llr(url, filepath)
        if model_type == 'weights':
            kwargs = get_pixelcnn_default_kwargs()
    detector = load_detector(filepath, **kwargs)
    return detector
