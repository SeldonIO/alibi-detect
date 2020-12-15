# type: ignore
# TODO: need to rewrite utilities using isinstance or @singledispatch for type checking to work properly
from functools import partial
import logging
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer
from tensorflow_probability.python.distributions.distribution import Distribution
from transformers import AutoTokenizer
from typing import Callable, Dict, List, Optional, Tuple, Union
from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.ad.adversarialae import DenseHidden
from alibi_detect.base import BaseDetector
from alibi_detect.cd import KSDrift, MMDDrift
from alibi_detect.cd.preprocess import HiddenOutput, UAE
from alibi_detect.models.autoencoder import AE, AEGMM, DecoderLSTM, EncoderLSTM, Seq2Seq, VAE, VAEGMM
from alibi_detect.models.embedding import TransformerEmbedding
from alibi_detect.models.pixelcnn import PixelCNN
from alibi_detect.od import (IForest, LLR, Mahalanobis, OutlierAE, OutlierAEGMM, OutlierProphet,
                             OutlierSeq2Seq, OutlierVAE, OutlierVAEGMM, SpectralResidual)
from alibi_detect.od.llr import build_model

logger = logging.getLogger(__name__)

Data = Union[
    BaseDetector,
    AdversarialAE,
    ModelDistillation,
    IForest,
    KSDrift,
    LLR,
    Mahalanobis,
    MMDDrift,
    OutlierAEGMM,
    OutlierAE,
    OutlierProphet,
    OutlierSeq2Seq,
    OutlierVAE,
    OutlierVAEGMM,
    SpectralResidual
]

DEFAULT_DETECTORS = [
    'AdversarialAE',
    'ModelDistillation',
    'IForest',
    'KSDrift',
    'LLR',
    'Mahalanobis',
    'MMDDrift',
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
    Save outlier, drift or adversarial detector.

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
    elif detector_name == 'KSDrift':
        state_dict, model, embed, embed_args, tokenizer = state_ksdrift(detector)
    elif detector_name == 'MMDDrift':
        state_dict, model, embed, embed_args, tokenizer = state_mmddrift(detector)
    elif detector_name == 'OutlierAEGMM':
        state_dict = state_aegmm(detector)
    elif detector_name == 'OutlierVAEGMM':
        state_dict = state_vaegmm(detector)
    elif detector_name == 'AdversarialAE':
        state_dict = state_adv_ae(detector)
    elif detector_name == 'ModelDistillation':
        state_dict = state_adv_md(detector)
    elif detector_name == 'OutlierProphet':
        state_dict = state_prophet(detector)
    elif detector_name == 'SpectralResidual':
        state_dict = state_sr(detector)
    elif detector_name == 'OutlierSeq2Seq':
        state_dict = state_s2s(detector)
    elif detector_name == 'LLR':
        state_dict = state_llr(detector)

    with open(os.path.join(filepath, detector_name + '.pickle'), 'wb') as f:
        pickle.dump(state_dict, f)

    # save outlier detector specific TensorFlow models
    if detector_name == 'OutlierAE':
        save_tf_ae(detector, filepath)
    elif detector_name == 'OutlierVAE':
        save_tf_vae(detector, filepath)
    elif detector_name in ['KSDrift', 'MMDDrift']:
        if model is not None:
            save_tf_model(model, filepath, model_name='encoder')
        if embed is not None:
            save_embedding(embed, embed_args, filepath)
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(filepath, 'model'))
    elif detector_name == 'OutlierAEGMM':
        save_tf_aegmm(detector, filepath)
    elif detector_name == 'OutlierVAEGMM':
        save_tf_vaegmm(detector, filepath)
    elif detector_name == 'AdversarialAE':
        save_tf_ae(detector, filepath)
        save_tf_model(detector.model, filepath)
        save_tf_hl(detector.model_hl, filepath)
    elif detector_name == 'ModelDistillation':
        save_tf_model(detector.distilled_model, filepath, model_name='distilled_model')
        save_tf_model(detector.model, filepath, model_name='model')
    elif detector_name == 'OutlierSeq2Seq':
        save_tf_s2s(detector, filepath)
    elif detector_name == 'LLR':
        save_tf_llr(detector, filepath)


def save_embedding(embed: tf.keras.Model,
                   embed_args: dict,
                   filepath: str,
                   save_dir: str = 'model',
                   model_name: str = 'embedding') -> None:
    """
    Save embeddings for text drift models.

    Parameters
    ----------
    embed
        Embedding model.
    embed_args
        Arguments for TransformerEmbedding module.
    filepath
        Save directory.
    save_dir
        Save folder.
    model_name
        Name of saved model.
    """
    # create folder for model weights
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, save_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    embed.save_pretrained(model_dir)
    with open(os.path.join(filepath, save_dir, model_name + '.pickle'), 'wb') as f:
        pickle.dump(embed_args, f)


def preprocess_step_drift(cd: Union[KSDrift, MMDDrift]) \
        -> Tuple[
            Optional[Callable], Dict, Optional[Union[tf.keras.Model, tf.keras.Sequential]],
            Optional[TransformerEmbedding], Dict, Optional[Callable], bool
        ]:
    # note: need to be able to pickle tokenizers other than transformers
    preprocess_fn, preprocess_kwargs = None, {}
    model, embed, embed_args, tokenizer, load_emb = None, None, {}, None, False
    if isinstance(cd.preprocess_fn, partial):
        preprocess_fn = cd.preprocess_fn.func
        for k, v in cd.preprocess_fn.keywords.items():
            if isinstance(v, UAE):
                if isinstance(v.encoder.layers[0], TransformerEmbedding):  # text drift
                    # embedding
                    embed = v.encoder.layers[0].model
                    embed_args = dict(
                        embedding_type=v.encoder.layers[0].emb_type,
                        layers=v.encoder.layers[0].hs_emb.keywords['layers']
                    )
                    load_emb = True

                    # preprocessing encoder
                    inputs = Input(shape=cd.input_shape, dtype=tf.int64)
                    v.encoder.call(inputs)
                    shape_enc = (v.encoder.layers[0].output.shape[-1],)
                    layers = [InputLayer(input_shape=shape_enc)] + v.encoder.layers[1:]
                    model = tf.keras.Sequential(layers)
                    _ = model(tf.zeros((1,) + shape_enc))
                else:
                    model = v.encoder
                preprocess_kwargs['model'] = 'UAE'
            elif isinstance(v, HiddenOutput):
                model = v.model
                preprocess_kwargs['model'] = 'HiddenOutput'
            elif isinstance(v, (tf.keras.Sequential, tf.keras.Model)):
                model = v
                preprocess_kwargs['model'] = 'custom'
            elif hasattr(v, '__module__'):
                if 'transformers.tokenization' in v.__module__:  # transformers tokenizer
                    tokenizer = v
                    preprocess_kwargs[k] = v.__module__
            else:
                preprocess_kwargs[k] = v
    return preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb


def state_ksdrift(cd: KSDrift) -> Tuple[
            Dict, Optional[Union[tf.keras.Model, tf.keras.Sequential]],
            Optional[TransformerEmbedding], Optional[Dict], Optional[Callable]
        ]:
    """
    K-S drift detector parameters to save.

    Parameters
    ----------
    cd
        Drift detection object.
    """
    preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb = \
        preprocess_step_drift(cd)
    state_dict = {
        'p_val': cd.p_val,
        'X_ref': cd.X_ref,
        'preprocess_X_ref': cd.preprocess_X_ref,
        'update_X_ref': cd.update_X_ref,
        'alternative': cd.alternative,
        'n': cd.n,
        'n_features': cd.n_features,
        'correction': cd.correction,
        'preprocess_fn': preprocess_fn,
        'preprocess_kwargs': preprocess_kwargs,
        'input_shape': cd.input_shape,
        'load_text_embedding': load_emb
    }
    return state_dict, model, embed, embed_args, tokenizer


def state_mmddrift(cd: MMDDrift) -> Tuple[
            Dict, Optional[Union[tf.keras.Model, tf.keras.Sequential]],
            Optional[TransformerEmbedding], Optional[Dict], Optional[Callable]
        ]:
    """
    MMD drift detector parameters to save.

    Parameters
    ----------
    cd
        Drift detection object.
    """
    preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb = \
        preprocess_step_drift(cd)
    state_dict = {
        'p_val': cd.p_val,
        'X_ref': cd.X_ref,
        'preprocess_X_ref': cd.preprocess_X_ref,
        'update_X_ref': cd.update_X_ref,
        'n': cd.n,
        'chunk_size': cd.chunk_size,
        'permutation_test': cd.permutation_test,
        'infer_sigma': cd.infer_sigma,
        'preprocess_fn': preprocess_fn,
        'preprocess_kwargs': preprocess_kwargs,
        'input_shape': cd.input_shape,
        'load_text_embedding': load_emb
    }
    return state_dict, model, embed, embed_args, tokenizer


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


def state_adv_ae(ad: AdversarialAE) -> Dict:
    """
    AdversarialAE parameters to save.

    Parameters
    ----------
    ad
        Adversarial detector object.
    """
    state_dict = {'threshold': ad.threshold,
                  'w_model_hl': ad.w_model_hl,
                  'temperature': ad.temperature,
                  'hidden_layer_kld': ad.hidden_layer_kld}
    return state_dict


def state_adv_md(md: ModelDistillation) -> Dict:
    """
    ModelDistillation parameters to save.

    Parameters
    ----------
    md
        ModelDistillation detector object.
    """
    state_dict = {'threshold': md.threshold,
                  'temperature': md.temperature,
                  'loss_type': md.loss_type}
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


def state_llr(od: LLR) -> Dict:
    """
    LLR parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {
        'threshold': od.threshold,
        'has_log_prob': od.has_log_prob,
        'sequential': od.sequential,
        'log_prob': od.log_prob
    }
    return state_dict


def save_tf_ae(detector: Union[OutlierAE, AdversarialAE],
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


def save_tf_vae(detector: OutlierVAE,
                filepath: str) -> None:
    """
    Save TensorFlow components of OutlierVAE.

    Parameters
    ----------
    detector
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
                  filepath: str,
                  save_dir: str = None,
                  model_name: str = 'model') -> None:
    """
    Save TensorFlow model.

    Parameters
    ----------
    model
        tf.keras.Model or tf.keras.Sequential.
    filepath
        Save directory.
    save_dir
        Save folder.
    model_name
        Name of saved model.
    """
    # create folder for model weights
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    if not save_dir:
        model_dir = os.path.join(filepath, 'model')
    else:
        model_dir = os.path.join(filepath, save_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # save classification model
    if isinstance(model, tf.keras.Model) or isinstance(model, tf.keras.Sequential):
        model.save(os.path.join(model_dir, model_name + '.h5'))
    else:
        logger.warning('No `tf.keras.Model` or `tf.keras.Sequential` detected. No model saved.')


def save_tf_llr(detector: LLR, filepath: str) -> None:
    """
    Save LLR TensorFlow models or distributions.

    Parameters
    ----------
    detector
        Outlier detector object.
    filepath
        Save directory.
    """
    if not os.path.isdir(filepath):
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if hasattr(detector, 'model_s') and hasattr(detector, 'model_b'):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        detector.model_s.save_weights(os.path.join(model_dir, 'model_s.h5'))
        detector.model_b.save_weights(os.path.join(model_dir, 'model_b.h5'))
    else:
        detector.dist_s.save(os.path.join(model_dir, 'model.h5'))
        if detector.dist_b is not None:
            detector.dist_b.save(os.path.join(model_dir, 'model_background.h5'))


def save_tf_hl(models: List[tf.keras.Model],
               filepath: str) -> None:
    """
    Save TensorFlow model weights.

    Parameters
    ----------
    models
        List with tf.keras models.
    filepath
        Save directory.
    """
    if isinstance(models, list):

        # create folder for model weights
        if not os.path.isdir(filepath):
            logger.warning('Directory {} does not exist and is now created.'.format(filepath))
            os.mkdir(filepath)
        model_dir = os.path.join(filepath, 'model')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        for i, m in enumerate(models):
            model_path = os.path.join(model_dir, 'model_hl_' + str(i) + '.ckpt')
            m.save_weights(model_path)


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


def load_detector(filepath: str, **kwargs) -> Data:
    """
    Load outlier, drift or adversarial detector.

    Parameters
    ----------
    filepath
        Load directory.

    Returns
    -------
    Loaded outlier or adversarial detector object.
    """
    if kwargs:
        k = list(kwargs.keys())
    else:
        k = []

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
    elif detector_name == 'AdversarialAE':
        ae = load_tf_ae(filepath)
        custom_objects = kwargs['custom_objects'] if 'custom_objects' in k else None
        model = load_tf_model(filepath, custom_objects=custom_objects)
        model_hl = load_tf_hl(filepath, model, state_dict)
        detector = init_ad_ae(state_dict, ae, model, model_hl)
    elif detector_name == 'ModelDistillation':
        md = load_tf_model(filepath, model_name='distilled_model')
        custom_objects = kwargs['custom_objects'] if 'custom_objects' in k else None
        model = load_tf_model(filepath, custom_objects=custom_objects)
        detector = init_ad_md(state_dict, md, model)
    elif detector_name == 'OutlierProphet':
        detector = init_od_prophet(state_dict)
    elif detector_name == 'SpectralResidual':
        detector = init_od_sr(state_dict)
    elif detector_name == 'OutlierSeq2Seq':
        seq2seq = load_tf_s2s(filepath, state_dict)
        detector = init_od_s2s(state_dict, seq2seq)
    elif detector_name in ['KSDrift', 'MMDDrift']:
        emb, tokenizer = None, None
        if state_dict['load_text_embedding']:
            emb, tokenizer = load_text_embed(filepath)
        model = load_tf_model(filepath, model_name='encoder')
        load_fn = init_cd_ksdrift if detector_name == 'KSDrift' else init_cd_mmddrift
        detector = load_fn(state_dict, model, emb, tokenizer, **kwargs)
    elif detector_name == 'LLR':
        models = load_tf_llr(filepath, **kwargs)
        detector = init_od_llr(state_dict, models)

    detector.meta = meta_dict
    return detector


def load_tf_model(filepath: str,
                  load_dir: str = None,
                  custom_objects: dict = None,
                  model_name: str = 'model') -> tf.keras.Model:
    """
    Load TensorFlow model.

    Parameters
    ----------
    filepath
        Saved model directory.
    load_dir
        Saved model folder.
    custom_objects
        Optional custom objects when loading the TensorFlow model.
    model_name
        Name of loaded model.

    Returns
    -------
    Loaded model.
    """
    if not load_dir:
        model_dir = os.path.join(filepath, 'model')
    else:
        model_dir = os.path.join(filepath, load_dir)
    if model_name + '.h5' not in [f for f in os.listdir(model_dir) if not f.startswith('.')]:
        logger.warning('No model found in {}.'.format(model_dir))
        return None
    model = tf.keras.models.load_model(os.path.join(model_dir, model_name + '.h5'), custom_objects=custom_objects)
    return model


def load_tf_hl(filepath: str, model: tf.keras.Model, state_dict: dict) -> List[tf.keras.Model]:
    """
    Load hidden layer models for AdversarialAE.

    Parameters
    ----------
    filepath
        Saved model directory.
    model
        tf.keras classification model.
    state_dict
        Dictionary containing the detector's parameters.

    Returns
    -------
    List with loaded tf.keras models.
    """
    model_dir = os.path.join(filepath, 'model')
    hidden_layer_kld = state_dict['hidden_layer_kld']
    if not hidden_layer_kld:
        return None
    model_hl = []
    for i, (hidden_layer, output_dim) in enumerate(hidden_layer_kld.items()):
        m = DenseHidden(model, hidden_layer, output_dim)
        m.load_weights(os.path.join(model_dir, 'model_hl_' + str(i) + '.ckpt'))
        model_hl.append(m)
    return model_hl


def load_tf_ae(filepath: str) -> tf.keras.Model:
    """
    Load AE.

    Parameters
    ----------
    filepath
        Save directory.

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


def load_tf_llr(filepath: str, dist_s: Union[Distribution, PixelCNN] = None,
                dist_b: Union[Distribution, PixelCNN] = None, input_shape: tuple = None):
    """
    Load LLR TensorFlow models or distributions.

    Parameters
    ----------
    detector
        Likelihood ratio detector.
    filepath
        Save directory.
    dist_s
        TensorFlow distribution for semantic model.
    dist_b
        TensorFlow distribution for background model.
    input_shape
        Input shape of the model.

    Returns
    -------
    Detector with loaded models.
    """
    model_dir = os.path.join(filepath, 'model')
    if 'model_s.h5' in os.listdir(model_dir) and 'model_b.h5' in os.listdir(model_dir):
        model_s, dist_s = build_model(dist_s, input_shape, os.path.join(model_dir, 'model_s.h5'))
        model_b, dist_b = build_model(dist_b, input_shape, os.path.join(model_dir, 'model_b.h5'))
        return dist_s, dist_b, model_s, model_b
    else:
        dist_s = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'), compile=False)
        if 'model_background.h5' in os.listdir(model_dir):
            dist_b = tf.keras.models.load_model(os.path.join(model_dir, 'model_background.h5'), compile=False)
        else:
            dist_b = None
        return dist_s, dist_b, None, None


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


def init_ad_ae(state_dict: Dict,
               ae: tf.keras.Model,
               model: tf.keras.Model,
               model_hl: List[tf.keras.Model]) -> AdversarialAE:
    """
    Initialize AdversarialAE.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    ae
        Loaded VAE.
    model
        Loaded classification model.
    model_hl
        List of tf.keras models.

    Returns
    -------
    Initialized AdversarialAE instance.
    """
    ad = AdversarialAE(threshold=state_dict['threshold'],
                       ae=ae,
                       model=model,
                       model_hl=model_hl,
                       w_model_hl=state_dict['w_model_hl'],
                       temperature=state_dict['temperature'])
    return ad


def init_ad_md(state_dict: Dict,
               distilled_model: tf.keras.Model,
               model: tf.keras.Model) -> ModelDistillation:
    """
    Initialize ModelDistillation.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    distilled_model
        Loaded distilled model.
    model
        Loaded classification model.

    Returns
    -------
    Initialized ModelDistillation instance.
    """
    ad = ModelDistillation(threshold=state_dict['threshold'],
                           distilled_model=distilled_model,
                           model=model,
                           temperature=state_dict['temperature'],
                           loss_type=state_dict['loss_type'])
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


def load_text_embed(filepath: str, load_dir: str = 'model') \
        -> Tuple[TransformerEmbedding, Callable]:
    model_dir = os.path.join(filepath, load_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    args = pickle.load(open(os.path.join(model_dir, 'embedding.pickle'), 'rb'))
    emb = TransformerEmbedding(
        model_dir, embedding_type=args['embedding_type'], layers=args['layers']
    )
    return emb, tokenizer


def init_preprocess(state_dict: Dict, model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
                    emb: Optional[TransformerEmbedding], tokenizer: Optional[Callable], **kwargs) \
        -> Tuple[Optional[Callable], Optional[dict]]:
    """ Return preprocessing function and kwargs. """
    if kwargs:  # override defaults
        keys = list(kwargs.keys())
        preprocess_fn = kwargs['preprocess_fn'] if 'preprocess_fn' in keys else None
        preprocess_kwargs = kwargs['preprocess_kwargs'] if 'preprocess_kwargs' in keys else None
        return preprocess_fn, preprocess_kwargs
    elif model is not None and isinstance(state_dict['preprocess_fn'], Callable) \
            and isinstance(state_dict['preprocess_kwargs'], dict):
        preprocess_fn = state_dict['preprocess_fn']
        preprocess_kwargs = state_dict['preprocess_kwargs']
    else:
        return None, None

    keys = list(preprocess_kwargs.keys())

    if 'model' not in keys:
        raise ValueError('No model found for the preprocessing step.')

    if preprocess_kwargs['model'] == 'UAE':
        if emb is not None:
            model = tf.keras.Sequential([emb] + model.layers)
            preprocess_kwargs['tokenizer'] = tokenizer
        preprocess_kwargs['model'] = UAE(encoder_net=model)
    else:  # incl. preprocess_kwargs['model'] == 'HiddenOutput'
        preprocess_kwargs['model'] = model

    return preprocess_fn, preprocess_kwargs


def init_cd_ksdrift(state_dict: Dict, model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
                    emb: Optional[TransformerEmbedding], tokenizer: Optional[Callable], **kwargs) \
        -> KSDrift:
    """
    Initialize KSDrift detector.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    model
        Optional preprocessing model.
    emb
        Optional text embedding model.
    tokenizer
        Optional tokenizer for text drift.
    kwargs
        Kwargs optionally containing preprocess_fn and preprocess_kwargs.

    Returns
    -------
    Initialized KSDrift instance.
    """
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict, model, emb, tokenizer, **kwargs)
    cd = KSDrift(
        p_val=state_dict['p_val'],
        X_ref=state_dict['X_ref'],
        preprocess_X_ref=False,
        update_X_ref=state_dict['update_X_ref'],
        preprocess_fn=preprocess_fn,
        preprocess_kwargs=preprocess_kwargs,
        correction=state_dict['correction'],
        alternative=state_dict['alternative'],
        n_features=state_dict['n_features'],
        input_shape=state_dict['input_shape']
    )
    cd.n = state_dict['n']
    cd.preprocess_X_ref = state_dict['preprocess_X_ref']
    return cd


def init_cd_mmddrift(state_dict: Dict, model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
                     emb: Optional[TransformerEmbedding], tokenizer: Optional[Callable], **kwargs) \
        -> MMDDrift:
    """
    Initialize MMDDrift detector.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.
    model
        Optional preprocessing model.
    emb
        Optional text embedding model.
    tokenizer
        Optional tokenizer for text drift.
    kwargs
        Kwargs optionally containing preprocess_fn and preprocess_kwargs.

    Returns
    -------
    Initialized MMDDrift instance.
    """
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict, model, emb, tokenizer, **kwargs)
    cd = MMDDrift(
        p_val=state_dict['p_val'],
        X_ref=state_dict['X_ref'],
        preprocess_X_ref=False,
        update_X_ref=state_dict['update_X_ref'],
        preprocess_fn=preprocess_fn,
        preprocess_kwargs=preprocess_kwargs,
        chunk_size=state_dict['chunk_size'],
        input_shape=state_dict['input_shape']
    )
    cd.n = state_dict['n']
    cd.preprocess_X_ref = state_dict['preprocess_X_ref']
    cd.infer_sigma = state_dict['infer_sigma']
    cd.permutation_test = state_dict['permutation_test']
    return cd


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


def init_od_llr(state_dict: Dict, models: tuple) -> LLR:
    """
    Initialize LLR detector.

    Parameters
    ----------
    state_dict
        Dictionary containing the parameter values.

    Returns
    -------
    Initialized LLR instance.
    """
    od = LLR(threshold=state_dict['threshold'],
             model=models[0],
             model_background=models[1],
             log_prob=state_dict['log_prob'],
             sequential=state_dict['sequential'])
    if models[2] is not None and models[3] is not None:
        od.model_s = models[2]
        od.model_b = models[3]
    return od
