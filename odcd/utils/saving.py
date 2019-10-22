import logging
import os
import pickle
from typing import Dict, Union
from odcd.od.base import BaseOutlierDetector
from odcd.od.vae import OutlierVAE

logger = logging.getLogger(__name__)

Data = Union[BaseOutlierDetector, OutlierVAE]

DEFAULT_OUTLIER_DETECTORS = ['OutlierVAE']


def save_od(od: Data,
            filepath: str) -> None:
    """
    Save outlier detector.

    Parameters
    ----------
    od
        Outlier detector object.
    filepath
        Save directory.
    """
    od_name = od.meta['name']
    if od_name not in DEFAULT_OUTLIER_DETECTORS:
        raise ValueError('{} is not supported by `save_od`.'.format(od_name))

    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)

    # save metadata
    with open(filepath + 'meta.pickle', 'wb') as f:
        pickle.dump(od.meta, f)

    # save outlier detector specific parameters
    if od_name == 'OutlierVAE':
        state_dict = state_vae(od)

    with open(filepath + od_name + '.pickle', 'wb') as f:
        pickle.dump(state_dict, f)

    # save outlier detector specific TensorFlow models
    if od_name == 'OutlierVAE':
        save_tf_vae(od, filepath)


def state_vae(od: OutlierVAE) -> Dict:
    """
    OutlierVAE parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'threshold': od.threshold,
                  'score_type': od.score_type,
                  'samples': od.samples,
                  'latent_dim': od.vae.latent_dim,
                  'beta': od.vae.beta}
    return state_dict


def save_tf_vae(od: OutlierVAE,
                filepath: str) -> None:
    """
    Save TensorFlow components of OutlierVAE.

    Parameters
    ----------
    od
        Outlier detector object.
    filepath
        Save directory.
    """
    # create folder for model weights
    model_dir = filepath + 'model/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # save encoder, decoder and vae weights
    od.vae.encoder.encoder_net.save(model_dir + 'encoder_net.h5')
    od.vae.decoder.decoder_net.save(model_dir + 'decoder_net.h5')
    od.vae.save_weights(model_dir + 'vae.ckpt')
