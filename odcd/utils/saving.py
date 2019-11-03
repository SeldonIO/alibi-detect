import logging
import os
import pickle
import tensorflow as tf
from typing import Dict, Union
from odcd.models.autoencoder import AEGMM, VAE
from odcd.od.aegmm import OutlierAEGMM
from odcd.od.base import BaseOutlierDetector
from odcd.od.isolationforest import IForest
from odcd.od.mahalanobis import Mahalanobis
from odcd.od.vae import OutlierVAE

logger = logging.getLogger(__name__)

Data = Union[BaseOutlierDetector,
             IForest,
             Mahalanobis,
             OutlierAEGMM,
             OutlierVAE]

DEFAULT_OUTLIER_DETECTORS = ['IForest',
                             'Mahalanobis',
                             'OutlierAEGMM',
                             'OutlierVAE']


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
    elif od_name == 'IForest':
        state_dict = state_iforest(od)
    elif od_name == 'OutlierAEGMM':
        state_dict = state_aegmm(od)

    with open(filepath + od_name + '.pickle', 'wb') as f:
        pickle.dump(state_dict, f)

    # save outlier detector specific TensorFlow models
    if od_name == 'OutlierVAE':
        save_tf_vae(od, filepath)
    elif od_name == 'OutlierAEGMM':
        save_tf_aegmm(od, filepath)


def state_iforest(od: IForest) -> Dict:
    """
    Isolation forest parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'threshold': od.threshold,
                  'isolationforest': od.isolationforest}
    return state_dict


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


def state_aegmm(od: OutlierAEGMM) -> Dict:
    """
    OutlierVAE parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'threshold': od.threshold,
                  'n_gmm': od.aegmm.n_gmm,
                  'recon_features': od.aegmm.recon_features,
                  'phi': od.phi,
                  'mu': od.mu,
                  'cov': od.cov,
                  'L': od.L,
                  'log_det_cov': od.log_det_cov}
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


def save_tf_aegmm(od: OutlierAEGMM,
                  filepath: str) -> None:
    """
    Save TensorFlow components of OutlierAEGMM.

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
    # save encoder, decoder, gmm density model and aegmm weights
    if isinstance(od.aegmm.encoder, tf.keras.Sequential):
        od.aegmm.encoder.save(model_dir + 'encoder_net.h5')
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(od.aegmm.decoder, tf.keras.Sequential):
        od.aegmm.decoder.save(model_dir + 'decoder_net.h5')
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(od.aegmm.gmm_density, tf.keras.Sequential):
        od.aegmm.gmm_density.save(model_dir + 'gmm_density_net.h5')
    else:
        logger.warning('No `tf.keras.Sequential` GMM density net detected. No GMM density net saved.')
    if isinstance(od.aegmm, tf.keras.Model):
        od.aegmm.save_weights(model_dir + 'aegmm.ckpt')
    else:
        logger.warning('No `tf.keras.Model` AEGMM detected. No AEGMM saved.')


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

    # initialize outlier detector
    if od_name == 'OutlierVAE':
        vae = load_tf_vae(filepath, state_dict)
        od = init_od_vae(state_dict, vae)
    elif od_name == 'Mahalanobis':
        od = init_od_mahalanobis(state_dict)
    elif od_name == 'IForest':
        od = init_od_iforest(state_dict)
    elif od_name == 'OutlierAEGMM':
        aegmm = load_tf_aegmm(filepath, state_dict)
        od = init_od_aegmm(state_dict, aegmm)

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


def load_tf_aegmm(filepath: str,
                  state_dict: Dict) -> tf.keras.Model:
    """
    Load AEGMM.

    Parameters
    ----------
    filepath
        Save directory.
    state_dict
        Dictionary containing the `n_gmm` and `recon_features` parameters.

    Returns
    -------
    Loaded AEGMM.
    """
    model_dir = filepath + 'model/'
    if not [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No encoder, decoder, gmm density net or aegmm found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir + 'encoder_net.h5')
    decoder_net = tf.keras.models.load_model(model_dir + 'decoder_net.h5')
    gmm_density_net = tf.keras.models.load_model(model_dir + 'gmm_density_net.h5')
    aegmm = AEGMM(encoder_net, decoder_net, gmm_density_net, state_dict['n_gmm'], state_dict['recon_features'])
    aegmm.load_weights(model_dir + 'aegmm.ckpt')
    return aegmm


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


def init_od_aegmm(state_dict: Dict,
                  aegmm: tf.keras.Model) -> OutlierAEGMM:
    """
    Initialize OutlierVAE.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    aegmm
        Loaded AEGMM.

    Returns
    -------
    Initialized OutlierAEGMM instance.
    """
    od = OutlierAEGMM(threshold=state_dict['threshold'],
                      aegmm=aegmm)
    od.phi = state_dict['phi']
    od.mu = state_dict['mu']
    od.cov = state_dict['cov']
    od.L = state_dict['L']
    od.log_det_cov = state_dict['log_det_cov']
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


def init_od_iforest(state_dict: Dict) -> IForest:
    """
    Initialize isolation forest.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized IForest instance.
    """
    od = IForest(threshold=state_dict['threshold'])
    od.isolationforest = state_dict['isolationforest']
    return od
