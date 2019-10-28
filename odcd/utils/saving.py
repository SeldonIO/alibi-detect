import logging
import os
import pickle
import tensorflow as tf
from typing import Dict, Union
from odcd.models.autoencoder import VAE
from odcd.od.base import BaseOutlierDetector
from odcd.od.vae import OutlierVAE
from odcd.od.mahalanobis import Mahalanobis

logger = logging.getLogger(__name__)

Data = Union[BaseOutlierDetector, OutlierVAE, Mahalanobis]

DEFAULT_OUTLIER_DETECTORS = ['OutlierVAE', 'Mahalanobis']


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
    elif od_name == 'Mahalanobis':
        state_dict = state_mahalanobis(od)

    with open(filepath + od_name + '.pickle', 'wb') as f:
        pickle.dump(state_dict, f)

    # save outlier detector specific TensorFlow models
    if od_name == 'OutlierVAE':
        save_tf_vae(od, filepath)


def state_mahalanobis(od: Mahalanobis) -> Dict:
    """
    Mahalanobis parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'threshold': od.threshold,
                  'n_components': od.n_components,
                  'std_clip': od.std_clip,
                  'start_clip': od.start_clip,
                  'max_n': od.max_n,
                  'cat_vars': od.cat_vars,
                  'ohe': od.ohe,
                  'd_abs': od.d_abs,
                  'clip': od.clip,
                  'mean': od.mean,
                  'C': od.C,
                  'n': od.n}
    return state_dict


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
    if isinstance(od.vae.encoder.encoder_net, tf.keras.Sequential):
        od.vae.encoder.encoder_net.save(model_dir + 'encoder_net.h5')
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(od.vae.decoder.decoder_net, tf.keras.Sequential):
        od.vae.decoder.decoder_net.save(model_dir + 'decoder_net.h5')
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(od.vae, tf.keras.Model):
        od.vae.save_weights(model_dir + 'vae.ckpt')
    else:
        logger.warning('No `tf.keras.Model` vae detected. No vae saved.')


def load_od(filepath: str) -> Data:
    """
    Load outlier detector.

    Parameters
    ----------
    filepath
        Load directory.

    Returns
    -------
    Loaded outlier detector object.
    """
    # check if path exists
    if not os.path.isdir(filepath):
        raise ValueError('{} does not exist.'.format(filepath))

    # load metadata
    meta_dict = pickle.load(open(filepath + 'meta.pickle', 'rb'))

    od_name = meta_dict['name']
    if od_name not in DEFAULT_OUTLIER_DETECTORS:
        raise ValueError('{} is not supported by `load_od`.'.format(od_name))

    # load outlier detector specific parameters
    state_dict = pickle.load(open(filepath + od_name + '.pickle', 'rb'))

    # load outlier detector specific TensorFlow models
    if od_name == 'OutlierVAE':
        vae = load_tf_vae(filepath, state_dict)

    # initialize outlier detector
    if od_name == 'OutlierVAE':
        od = init_od_vae(state_dict, vae)
    elif od_name == 'Mahalanobis':
        od = init_od_mahalanobis(state_dict)

    od.meta = meta_dict
    return od


def load_tf_vae(filepath: str,
                state_dict: Dict) -> tf.keras.Model:
    """
    Load VAE.

    Parameters
    ----------
    filepath
        Save directory.
    state_dict
        Dictionary containing the latent dimension and beta parameters.

    Returns
    -------
    Loaded VAE.
    """
    model_dir = filepath + 'model/'
    if not [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No encoder, decoder or vae found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir + 'encoder_net.h5')
    decoder_net = tf.keras.models.load_model(model_dir + 'decoder_net.h5')
    vae = VAE(encoder_net, decoder_net, state_dict['latent_dim'], beta=state_dict['beta'])
    vae.load_weights(model_dir + 'vae.ckpt')
    return vae


def init_od_vae(state_dict: Dict,
                vae: tf.keras.Model) -> OutlierVAE:
    """
    Initialize OutlierVAE.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    vae
        Loaded VAE.

    Returns
    -------
    Initialized OutlierVAE instance.
    """
    od = OutlierVAE(threshold=state_dict['threshold'],
                    score_type=state_dict['score_type'],
                    vae=vae,
                    samples=state_dict['samples'])
    return od


def init_od_mahalanobis(state_dict: Dict) -> Mahalanobis:
    """
    Initialize Mahalanobis.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized Mahalanobis instance.
    """
    od = Mahalanobis(threshold=state_dict['threshold'],
                     n_components=state_dict['n_components'],
                     std_clip=state_dict['std_clip'],
                     start_clip=state_dict['start_clip'],
                     max_n=state_dict['max_n'],
                     cat_vars=state_dict['cat_vars'],
                     ohe=state_dict['ohe'])
    od.d_abs = state_dict['d_abs']
    od.clip = state_dict['clip']
    od.mean = state_dict['mean']
    od.C = state_dict['C']
    od.n = state_dict['n']
    return od
