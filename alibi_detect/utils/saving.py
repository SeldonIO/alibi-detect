# type: ignore
# TODO: need to rewrite utilities using isinstance or @singledispatch for type checking to work properly

import logging
import os
import pickle
import tensorflow as tf
from typing import Dict, Union
from alibi_detect.ad import AdversarialVAE
from alibi_detect.base import BaseDetector
from alibi_detect.models.autoencoder import AE, AEGMM, DecoderLSTM, EncoderLSTM, Seq2Seq, VAE, VAEGMM
from alibi_detect.od import (IForest, Mahalanobis, OutlierAE, OutlierAEGMM, OutlierProphet,
                             OutlierSeq2Seq, OutlierVAE, OutlierVAEGMM, SpectralResidual)

logger = logging.getLogger(__name__)

Data = Union[
    BaseDetector,
    AdversarialVAE,
    IForest,
    Mahalanobis,
    OutlierAEGMM,
    OutlierAE,
    OutlierProphet,
    OutlierSeq2Seq,
    OutlierVAE,
    OutlierVAEGMM,
    SpectralResidual
]

DEFAULT_DETECTORS = [
    'AdversarialVAE',
    'IForest',
    'Mahalanobis',
    'OutlierAE',
    'OutlierAEGMM',
    'OutlierProphet',
    'OutlierSeq2Seq',
    'OutlierVAE',
    'OutlierVAEGMM',
    'SpectralResidual'
]


def save_detector(detector: Data,
                  filepath: str) -> None:
    """
    Save outlier or adversarial detector.

    Parameters
    ----------
    detector
        Detector object.
    filepath
        Save directory.
    """
    detector_name = detector.meta['name']
    if detector_name not in DEFAULT_DETECTORS:
        raise ValueError('{} is not supported by `save_detector`.'.format(detector_name))

    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)

    # save metadata
    with open(os.path.join(filepath, 'meta.pickle'), 'wb') as f:
        pickle.dump(detector.meta, f)

    # save outlier detector specific parameters
    if detector_name == 'OutlierAE':
        state_dict = state_ae(detector)
    elif detector_name == 'OutlierVAE':
        state_dict = state_vae(detector)
    elif detector_name == 'Mahalanobis':
        state_dict = state_mahalanobis(detector)
    elif detector_name == 'IForest':
        state_dict = state_iforest(detector)
    elif detector_name == 'OutlierAEGMM':
        state_dict = state_aegmm(detector)
    elif detector_name == 'OutlierVAEGMM':
        state_dict = state_vaegmm(detector)
    elif detector_name == 'AdversarialVAE':
        state_dict = state_adv_vae(detector)
    elif detector_name == 'OutlierProphet':
        state_dict = state_prophet(detector)
    elif detector_name == 'SpectralResidual':
        state_dict = state_sr(detector)
    elif detector_name == 'OutlierSeq2Seq':
        state_dict = state_s2s(detector)

    with open(os.path.join(filepath, detector_name + '.pickle'), 'wb') as f:
        pickle.dump(state_dict, f)

    # save outlier detector specific TensorFlow models
    if detector_name == 'OutlierAE':
        save_tf_ae(detector, filepath)
    elif detector_name == 'OutlierVAE':
        save_tf_vae(detector, filepath)
    elif detector_name == 'OutlierAEGMM':
        save_tf_aegmm(detector, filepath)
    elif detector_name == 'OutlierVAEGMM':
        save_tf_vaegmm(detector, filepath)
    elif detector_name == 'AdversarialVAE':
        save_tf_vae(detector, filepath)
        save_tf_model(detector.model, filepath)
    elif detector_name == 'OutlierSeq2Seq':
        save_tf_s2s(detector, filepath)


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


def state_ae(od: OutlierAE) -> Dict:
    """
    OutlierAE parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'threshold': od.threshold}
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
    OutlierAEGMM parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    if not all(tf.is_tensor(_) for _ in [od.phi, od.mu, od.cov, od.L, od.log_det_cov]):
        logger.warning('Saving AEGMM detector that has not been fit.')

    state_dict = {'threshold': od.threshold,
                  'n_gmm': od.aegmm.n_gmm,
                  'recon_features': od.aegmm.recon_features,
                  'phi': od.phi,
                  'mu': od.mu,
                  'cov': od.cov,
                  'L': od.L,
                  'log_det_cov': od.log_det_cov}
    return state_dict


def state_vaegmm(od: OutlierVAEGMM) -> Dict:
    """
    OutlierVAEGMM parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    if not all(tf.is_tensor(_) for _ in [od.phi, od.mu, od.cov, od.L, od.log_det_cov]):
        logger.warning('Saving VAEGMM detector that has not been fit.')

    state_dict = {'threshold': od.threshold,
                  'samples': od.samples,
                  'n_gmm': od.vaegmm.n_gmm,
                  'latent_dim': od.vaegmm.latent_dim,
                  'beta': od.vaegmm.beta,
                  'recon_features': od.vaegmm.recon_features,
                  'phi': od.phi,
                  'mu': od.mu,
                  'cov': od.cov,
                  'L': od.L,
                  'log_det_cov': od.log_det_cov}
    return state_dict


def state_adv_vae(ad: AdversarialVAE) -> Dict:
    """
    AdversarialVAE parameters to save.

    Parameters
    ----------
    ad
        Adversarial detector object.
    """
    state_dict = {'threshold': ad.threshold,
                  'samples': ad.samples,
                  'latent_dim': ad.vae.latent_dim,
                  'beta': ad.vae.beta}
    return state_dict


def state_prophet(od: OutlierProphet) -> Dict:
    """
    OutlierProphet parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'model': od.model,
                  'cap': od.cap}
    return state_dict


def state_sr(od: SpectralResidual) -> Dict:
    """
    Spectral residual parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'threshold': od.threshold,
                  'window_amp': od.window_amp,
                  'window_local': od.window_local,
                  'n_est_points': od.n_est_points,
                  'n_grad_points': od.n_grad_points}
    return state_dict


def state_s2s(od: OutlierSeq2Seq) -> Dict:
    """
    OutlierSeq2Seq parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'threshold': od.threshold,
                  'beta': od.seq2seq.beta,
                  'shape': od.shape,
                  'latent_dim': od.latent_dim,
                  'output_activation': od.output_activation}
    return state_dict


def save_tf_ae(detector: OutlierAE,
               filepath: str) -> None:
    """
    Save TensorFlow components of OutlierAE

    Parameters
    ----------
    detector
        Outlier or adversarial detector object.
    filepath
        Save directory.
    """
    # create folder for model weights
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # save encoder, decoder and vae weights
    if isinstance(detector.ae.encoder.encoder_net, tf.keras.Sequential):
        detector.ae.encoder.encoder_net.save(os.path.join(model_dir, 'encoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(detector.ae.decoder.decoder_net, tf.keras.Sequential):
        detector.ae.decoder.decoder_net.save(os.path.join(model_dir, 'decoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(detector.ae, tf.keras.Model):
        detector.ae.save_weights(os.path.join(model_dir, 'ae.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` ae detected. No ae saved.')


def save_tf_vae(detector: Union[OutlierVAE, AdversarialVAE],
                filepath: str) -> None:
    """
    Save TensorFlow components of OutlierVAE or AdversarialVAE.

    Parameters
    ----------
    detector
        Outlier or adversarial detector object.
    filepath
        Save directory.
    """
    # create folder for model weights
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # save encoder, decoder and vae weights
    if isinstance(detector.vae.encoder.encoder_net, tf.keras.Sequential):
        detector.vae.encoder.encoder_net.save(os.path.join(model_dir, 'encoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(detector.vae.decoder.decoder_net, tf.keras.Sequential):
        detector.vae.decoder.decoder_net.save(os.path.join(model_dir, 'decoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(detector.vae, tf.keras.Model):
        detector.vae.save_weights(os.path.join(model_dir, 'vae.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` vae detected. No vae saved.')


def save_tf_model(model: tf.keras.Model,
                  filepath: str) -> None:
    """
    Save TensorFlow model.

    Parameters
    ----------
    model
        A tf.keras Model.
    filepath
        Save directory.
    """
    # create folder for model weights
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # save classification model
    if isinstance(model, tf.keras.Model):  # TODO: not flexible enough!
        model.save(os.path.join(model_dir, 'model.h5'))
    else:
        logger.warning('No `tf.keras.Model` vae detected. No classification model saved.')


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
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # save encoder, decoder, gmm density model and aegmm weights
    if isinstance(od.aegmm.encoder, tf.keras.Sequential):
        od.aegmm.encoder.save(os.path.join(model_dir, 'encoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(od.aegmm.decoder, tf.keras.Sequential):
        od.aegmm.decoder.save(os.path.join(model_dir, 'decoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(od.aegmm.gmm_density, tf.keras.Sequential):
        od.aegmm.gmm_density.save(os.path.join(model_dir, 'gmm_density_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` GMM density net detected. No GMM density net saved.')
    if isinstance(od.aegmm, tf.keras.Model):
        od.aegmm.save_weights(os.path.join(model_dir, 'aegmm.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` AEGMM detected. No AEGMM saved.')


def save_tf_vaegmm(od: OutlierVAEGMM,
                   filepath: str) -> None:
    """
    Save TensorFlow components of OutlierVAEGMM.

    Parameters
    ----------
    od
        Outlier detector object.
    filepath
        Save directory.
    """
    # create folder for model weights
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # save encoder, decoder, gmm density model and vaegmm weights
    if isinstance(od.vaegmm.encoder.encoder_net, tf.keras.Sequential):
        od.vaegmm.encoder.encoder_net.save(os.path.join(model_dir, 'encoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(od.vaegmm.decoder, tf.keras.Sequential):
        od.vaegmm.decoder.save(os.path.join(model_dir, 'decoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(od.vaegmm.gmm_density, tf.keras.Sequential):
        od.vaegmm.gmm_density.save(os.path.join(model_dir, 'gmm_density_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` GMM density net detected. No GMM density net saved.')
    if isinstance(od.vaegmm, tf.keras.Model):
        od.vaegmm.save_weights(os.path.join(model_dir, 'vaegmm.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` VAEGMM detected. No VAEGMM saved.')


def save_tf_s2s(od: OutlierSeq2Seq,
                filepath: str) -> None:
    """
    Save TensorFlow components of OutlierSeq2Seq.

    Parameters
    ----------
    od
        Outlier detector object.
    filepath
        Save directory.
    """
    # create folder for model weights
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # save seq2seq model weights and threshold estimation network
    if isinstance(od.seq2seq.threshold_net, tf.keras.Sequential):
        od.seq2seq.threshold_net.save(os.path.join(model_dir, 'threshold_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` threshold estimation net detected. No threshold net saved.')
    if isinstance(od.seq2seq, tf.keras.Model):
        od.seq2seq.save_weights(os.path.join(model_dir, 'seq2seq.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` Seq2Seq detected. No Seq2Seq model saved.')


def load_detector(filepath: str) -> Data:
    """
    Load outlier or adversarial detector.

    Parameters
    ----------
    filepath
        Load directory.

    Returns
    -------
    Loaded outlier or adversarial detector object.
    """
    # check if path exists
    if not os.path.isdir(filepath):
        raise ValueError('{} does not exist.'.format(filepath))

    # load metadata
    meta_dict = pickle.load(open(os.path.join(filepath, 'meta.pickle'), 'rb'))

    detector_name = meta_dict['name']
    if detector_name not in DEFAULT_DETECTORS:
        raise ValueError('{} is not supported by `load_detector`.'.format(detector_name))

    # load outlier detector specific parameters
    state_dict = pickle.load(open(os.path.join(filepath, detector_name + '.pickle'), 'rb'))

    # initialize outlier detector
    if detector_name == 'OutlierAE':
        ae = load_tf_ae(filepath)
        detector = init_od_ae(state_dict, ae)
    elif detector_name == 'OutlierVAE':
        vae = load_tf_vae(filepath, state_dict)
        detector = init_od_vae(state_dict, vae)
    elif detector_name == 'Mahalanobis':
        detector = init_od_mahalanobis(state_dict)
    elif detector_name == 'IForest':
        detector = init_od_iforest(state_dict)
    elif detector_name == 'OutlierAEGMM':
        aegmm = load_tf_aegmm(filepath, state_dict)
        detector = init_od_aegmm(state_dict, aegmm)
    elif detector_name == 'OutlierVAEGMM':
        vaegmm = load_tf_vaegmm(filepath, state_dict)
        detector = init_od_vaegmm(state_dict, vaegmm)
    elif detector_name == 'AdversarialVAE':
        vae = load_tf_vae(filepath, state_dict)
        model = load_tf_model(filepath)
        detector = init_ad_vae(state_dict, vae, model)
    elif detector_name == 'OutlierProphet':
        detector = init_od_prophet(state_dict)
    elif detector_name == 'SpectralResidual':
        detector = init_od_sr(state_dict)
    elif detector_name == 'OutlierSeq2Seq':
        seq2seq = load_tf_s2s(filepath, state_dict)
        detector = init_od_s2s(state_dict, seq2seq)

    detector.meta = meta_dict
    return detector


def load_tf_model(filepath: str) -> tf.keras.Model:
    model_dir = os.path.join(filepath, 'model')
    if 'model.h5' not in [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No model found in {}.'.format(model_dir))
        return None
    model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
    return model


def load_tf_ae(filepath: str) -> tf.keras.Model:
    """
    Load AE.

    Parameters
    ----------
    filepath
        Save directory.
    state_dict
        Dictionary containing the outlier threshold.

    Returns
    -------
    Loaded AE.
    """
    model_dir = os.path.join(filepath, 'model')
    if not [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No encoder, decoder or ae found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'decoder_net.h5'))
    ae = AE(encoder_net, decoder_net)
    ae.load_weights(os.path.join(model_dir, 'ae.ckpt'))
    return ae


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
    model_dir = os.path.join(filepath, 'model')
    if not [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No encoder, decoder or vae found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'decoder_net.h5'))
    vae = VAE(encoder_net, decoder_net, state_dict['latent_dim'], beta=state_dict['beta'])
    vae.load_weights(os.path.join(model_dir, 'vae.ckpt'))
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
    model_dir = os.path.join(filepath, 'model')
    if not [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No encoder, decoder, gmm density net or aegmm found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'decoder_net.h5'))
    gmm_density_net = tf.keras.models.load_model(os.path.join(model_dir, 'gmm_density_net.h5'))
    aegmm = AEGMM(encoder_net, decoder_net, gmm_density_net, state_dict['n_gmm'], state_dict['recon_features'])
    aegmm.load_weights(os.path.join(model_dir, 'aegmm.ckpt'))
    return aegmm


def load_tf_vaegmm(filepath: str,
                   state_dict: Dict) -> tf.keras.Model:
    """
    Load VAEGMM.

    Parameters
    ----------
    filepath
        Save directory.
    state_dict
        Dictionary containing the `n_gmm`, `latent_dim` and `recon_features` parameters.

    Returns
    -------
    Loaded VAEGMM.
    """
    model_dir = os.path.join(filepath, 'model')
    if not [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No encoder, decoder, gmm density net or vaegmm found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'decoder_net.h5'))
    gmm_density_net = tf.keras.models.load_model(os.path.join(model_dir, 'gmm_density_net.h5'))
    vaegmm = VAEGMM(encoder_net, decoder_net, gmm_density_net, state_dict['n_gmm'],
                    state_dict['latent_dim'], state_dict['recon_features'], state_dict['beta'])
    vaegmm.load_weights(os.path.join(model_dir, 'vaegmm.ckpt'))
    return vaegmm


def load_tf_s2s(filepath: str,
                state_dict: Dict) -> tf.keras.Model:
    model_dir = os.path.join(filepath, 'model')
    if not [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No seq2seq or threshold estimation net found in {}.'.format(model_dir))
        return None
    # load threshold estimator net, initialize encoder and decoder and load seq2seq weights
    threshold_net = tf.keras.models.load_model(os.path.join(model_dir, 'threshold_net.h5'), compile=False)
    latent_dim = state_dict['latent_dim']
    n_features = state_dict['shape'][-1]
    encoder_net = EncoderLSTM(latent_dim)
    decoder_net = DecoderLSTM(latent_dim, n_features, state_dict['output_activation'])
    seq2seq = Seq2Seq(encoder_net, decoder_net, threshold_net, n_features, beta=state_dict['beta'])
    seq2seq.load_weights(os.path.join(model_dir, 'seq2seq.ckpt'))
    return seq2seq


def init_od_ae(state_dict: Dict,
               ae: tf.keras.Model) -> OutlierAE:
    """
    Initialize OutlierVAE.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    ae
        Loaded AE.

    Returns
    -------
    Initialized OutlierAE instance.
    """
    od = OutlierAE(threshold=state_dict['threshold'], ae=ae)
    return od


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


def init_ad_vae(state_dict: Dict,
                vae: tf.keras.Model,
                model: tf.keras.Model) -> AdversarialVAE:
    """
    Initialize AdversarialVAE.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    vae
        Loaded VAE.
    model
        Loaded classification model.

    Returns
    -------
    Initialized AdversarialVAE instance.
    """
    ad = AdversarialVAE(threshold=state_dict['threshold'],
                        vae=vae,
                        model=model,
                        samples=state_dict['samples'])
    return ad


def init_od_aegmm(state_dict: Dict,
                  aegmm: tf.keras.Model) -> OutlierAEGMM:
    """
    Initialize OutlierAEGMM.

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

    if not all(tf.is_tensor(_) for _ in [od.phi, od.mu, od.cov, od.L, od.log_det_cov]):
        logger.warning('Loaded AEGMM detector has not been fit.')

    return od


def init_od_vaegmm(state_dict: Dict,
                   vaegmm: tf.keras.Model) -> OutlierVAEGMM:
    """
    Initialize OutlierVAEGMM.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    vaegmm
        Loaded VAEGMM.

    Returns
    -------
    Initialized OutlierVAEGMM instance.
    """
    od = OutlierVAEGMM(threshold=state_dict['threshold'],
                       vaegmm=vaegmm,
                       samples=state_dict['samples'])
    od.phi = state_dict['phi']
    od.mu = state_dict['mu']
    od.cov = state_dict['cov']
    od.L = state_dict['L']
    od.log_det_cov = state_dict['log_det_cov']

    if not all(tf.is_tensor(_) for _ in [od.phi, od.mu, od.cov, od.L, od.log_det_cov]):
        logger.warning('Loaded VAEGMM detector has not been fit.')

    return od


def init_od_s2s(state_dict: Dict,
                seq2seq: tf.keras.Model) -> OutlierSeq2Seq:
    """
    Initialize OutlierSeq2Seq.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    seq2seq
        Loaded seq2seq model.

    Returns
    -------
    Initialized OutlierSeq2Seq instance.
    """
    seq_len, n_features = state_dict['shape'][1:]
    od = OutlierSeq2Seq(n_features,
                        seq_len,
                        threshold=state_dict['threshold'],
                        seq2seq=seq2seq,
                        latent_dim=state_dict['latent_dim'],
                        output_activation=state_dict['output_activation'])

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


def init_od_prophet(state_dict: Dict) -> OutlierProphet:
    """
    Initialize OutlierProphet.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized OutlierProphet instance.
    """
    od = OutlierProphet(cap=state_dict['cap'])
    od.model = state_dict['model']
    return od


def init_od_sr(state_dict: Dict) -> SpectralResidual:
    """
    Initialize spectral residual detector.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized SpectralResidual instance.
    """
    od = SpectralResidual(threshold=state_dict['threshold'],
                          window_amp=state_dict['window_amp'],
                          window_local=state_dict['window_local'],
                          n_est_points=state_dict['n_est_points'],
                          n_grad_points=state_dict['n_grad_points'])
    return od
