# type: ignore
# TODO: need to rewrite utilities using isinstance or @singledispatch for type checking to work properly
import dill
from functools import partial
import logging
import os
from pathlib import Path
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer
from tensorflow_probability.python.distributions.distribution import Distribution
from transformers import AutoTokenizer
from typing import Callable, Dict, List, Optional, Tuple, Union
from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.ad.adversarialae import DenseHidden
from alibi_detect.base import BaseDetector
from alibi_detect.cd import ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, TabularDrift
from alibi_detect.cd.tensorflow.classifier import ClassifierDriftTF
from alibi_detect.cd.tensorflow.mmd import MMDDriftTF
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.cd.tensorflow.preprocess import _Encoder
from alibi_detect.models.tensorflow.autoencoder import AE, AEGMM, DecoderLSTM, EncoderLSTM, Seq2Seq, VAE, VAEGMM
from alibi_detect.models.tensorflow import PixelCNN, TransformerEmbedding
from alibi_detect.od import (IForest, LLR, Mahalanobis, OutlierAE, OutlierAEGMM, OutlierProphet,
                             OutlierSeq2Seq, OutlierVAE, OutlierVAEGMM, SpectralResidual)
from alibi_detect.od.llr import build_model
from alibi_detect.utils.tensorflow.kernels import GaussianRBF
from alibi_detect.version import __version__

# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

logger = logging.getLogger(__name__)

Data = Union[
    AdversarialAE,
    BaseDetector,
    ChiSquareDrift,
    ClassifierDrift,
    ClassifierDriftTF,
    IForest,
    KSDrift,
    LLR,
    Mahalanobis,
    MMDDrift,
    MMDDriftTF,
    ModelDistillation,
    OutlierAE,
    OutlierAEGMM,
    OutlierProphet,
    OutlierSeq2Seq,
    OutlierVAE,
    OutlierVAEGMM,
    SpectralResidual,
    TabularDrift
]

DEFAULT_DETECTORS = [
    'AdversarialAE',
    'ChiSquareDrift',
    'ClassifierDrift',
    'ClassifierDriftTF',
    'IForest',
    'KSDrift',
    'LLR',
    'Mahalanobis',
    'MMDDrift',
    'MMDDriftTF',
    'ModelDistillation',
    'OutlierAE',
    'OutlierAEGMM',
    'OutlierProphet',
    'OutlierSeq2Seq',
    'OutlierVAE',
    'OutlierVAEGMM',
    'SpectralResidual',
    'TabularDrift'
]


def save_detector(detector: Data, filepath: Union[str, os.PathLike]) -> None:
    """
    Save outlier, drift or adversarial detector.

    Parameters
    ----------
    detector
        Detector object.
    filepath
        Save directory.
    """
    if 'backend' in list(detector.meta.keys()) and detector.meta['backend'] == 'pytorch':
        raise NotImplementedError('Detectors with PyTorch backend are not yet supported.')

    detector_name = detector.meta['name']
    if detector_name not in DEFAULT_DETECTORS:
        raise ValueError('{} is not supported by `save_detector`.'.format(detector_name))

    # check if path exists
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # save metadata
    with open(filepath.joinpath('meta.dill'), 'wb') as f:
        dill.dump(detector.meta, f)

    # save outlier detector specific parameters
    if detector_name == 'OutlierAE':
        state_dict = state_ae(detector)
    elif detector_name == 'OutlierVAE':
        state_dict = state_vae(detector)
    elif detector_name == 'Mahalanobis':
        state_dict = state_mahalanobis(detector)
    elif detector_name == 'IForest':
        state_dict = state_iforest(detector)
    elif detector_name == 'ChiSquareDrift':
        state_dict, model, embed, embed_args, tokenizer = state_chisquaredrift(detector)
    elif detector_name == 'ClassifierDriftTF':
        state_dict, clf_drift, model, embed, embed_args, tokenizer = state_classifierdrift(detector)
    elif detector_name == 'TabularDrift':
        state_dict, model, embed, embed_args, tokenizer = state_tabulardrift(detector)
    elif detector_name == 'KSDrift':
        state_dict, model, embed, embed_args, tokenizer = state_ksdrift(detector)
    elif detector_name == 'MMDDriftTF':
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

    with open(filepath.joinpath(detector_name + '.dill'), 'wb') as f:
        dill.dump(state_dict, f)

    # save outlier detector specific TensorFlow models
    if detector_name == 'OutlierAE':
        save_tf_ae(detector, filepath)
    elif detector_name == 'OutlierVAE':
        save_tf_vae(detector, filepath)
    elif detector_name in ['ChiSquareDrift', 'ClassifierDriftTF', 'KSDrift', 'MMDDriftTF', 'TabularDrift']:
        if model is not None:
            save_tf_model(model, filepath, model_name='encoder')
        if embed is not None:
            save_embedding(embed, embed_args, filepath)
        if tokenizer is not None:
            tokenizer.save_pretrained(filepath.joinpath('model'))
        if detector_name == 'ClassifierDriftTF':
            save_tf_model(clf_drift, filepath, model_name='clf_drift')
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
                   filepath: Union[str, os.PathLike],
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
        Name of folder to save to within the filepath directory.
    model_name
        Name of saved model.
    """
    # create folder to save model in
    model_dir = Path(filepath).joinpath(save_dir)
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)

    # Save embedding model
    embed.save_pretrained(model_dir)
    with open(model_dir.joinpath(model_name + '.dill'), 'wb') as f:
        dill.dump(embed_args, f)


def preprocess_step_drift(cd: Union[ChiSquareDrift, ClassifierDriftTF, KSDrift, MMDDriftTF, TabularDrift]) \
        -> Tuple[
            Optional[Callable], Dict, Optional[Union[tf.keras.Model, tf.keras.Sequential]],
            Optional[TransformerEmbedding], Dict, Optional[Callable], bool
        ]:
    # note: need to be able to dill tokenizers other than transformers
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
                if 'transformers' in v.__module__:  # transformers tokenizer
                    tokenizer = v
                    preprocess_kwargs[k] = v.__module__
            else:
                preprocess_kwargs[k] = v
    elif isinstance(cd.preprocess_fn, Callable):
        preprocess_fn = cd.preprocess_fn
    return preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb


def state_chisquaredrift(cd: ChiSquareDrift) -> Tuple[
            Dict, Optional[Union[tf.keras.Model, tf.keras.Sequential]],
            Optional[TransformerEmbedding], Optional[Dict], Optional[Callable]
        ]:
    """
    Chi-Squared drift detector parameters to save.

    Parameters
    ----------
    cd
        Drift detection object.
    """
    preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb = \
        preprocess_step_drift(cd)
    state_dict = {
        'args':
            {
                'x_ref': cd.x_ref
            },
        'kwargs':
            {
                'p_val': cd.p_val,
                'categories_per_feature': cd.x_ref_categories,
                'preprocess_x_ref': False,
                'update_x_ref': cd.update_x_ref,
                'correction': cd.correction,
                'n_features': cd.n_features,
                'input_shape': cd.input_shape,
            },
        'other':
            {
                'n': cd.n,
                'preprocess_x_ref': cd.preprocess_x_ref,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
    }
    return state_dict, model, embed, embed_args, tokenizer


def state_classifierdrift(cd: ClassifierDrift) -> Tuple[
            Dict, Union[tf.keras.Sequential, tf.keras.Model],
            Optional[Union[tf.keras.Model, tf.keras.Sequential]],
            Optional[TransformerEmbedding], Optional[Dict], Optional[Callable]
        ]:
    """
    Classifier-based drift detector parameters to save.

    Parameters
    ----------
    cd
        Drift detection object.
    """
    preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb = \
        preprocess_step_drift(cd._detector)
    cd._detector.train_kwargs['optimizer'] = tf.keras.optimizers.serialize(cd._detector.train_kwargs['optimizer'])
    state_dict = {
        'args':
            {
                'x_ref': cd._detector.x_ref,
            },
        'kwargs':
            {
                'p_val': cd._detector.p_val,
                'preprocess_x_ref': False,
                'update_x_ref': cd._detector.update_x_ref,
                'preds_type': cd._detector.preds_type,
                'binarize_preds': cd._detector.binarize_preds,
                'train_size': cd._detector.train_size,
                'train_kwargs': cd._detector.train_kwargs,
            },
        'other':
            {
                'n': cd._detector.n,
                'preprocess_x_ref': cd._detector.preprocess_x_ref,
                'skf': cd._detector.skf,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
    }
    return state_dict, cd._detector.model, model, embed, embed_args, tokenizer


def state_tabulardrift(cd: TabularDrift) -> Tuple[
            Dict, Optional[Union[tf.keras.Model, tf.keras.Sequential]],
            Optional[TransformerEmbedding], Optional[Dict], Optional[Callable]
        ]:
    """
    Tabular drift detector parameters to save.

    Parameters
    ----------
    cd
        Drift detection object.
    """
    preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb = \
        preprocess_step_drift(cd)
    state_dict = {
        'args':
            {
                'x_ref': cd.x_ref
            },
        'kwargs':
            {
                'p_val': cd.p_val,
                'categories_per_feature': cd.x_ref_categories,
                'preprocess_x_ref': False,
                'update_x_ref': cd.update_x_ref,
                'correction': cd.correction,
                'alternative': cd.alternative,
                'n_features': cd.n_features,
                'input_shape': cd.input_shape,
            },
        'other':
            {
                'n': cd.n,
                'preprocess_x_ref': cd.preprocess_x_ref,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
    }
    return state_dict, model, embed, embed_args, tokenizer


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
        'args':
            {
                'x_ref': cd.x_ref
            },
        'kwargs':
            {
                'p_val': cd.p_val,
                'preprocess_x_ref': False,
                'update_x_ref': cd.update_x_ref,
                'correction': cd.correction,
                'alternative': cd.alternative,
                'n_features': cd.n_features,
                'input_shape': cd.input_shape,
            },
        'other':
            {
                'n': cd.n,
                'preprocess_x_ref': cd.preprocess_x_ref,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
    }
    return state_dict, model, embed, embed_args, tokenizer


def state_mmddrift(cd: MMDDrift) -> Tuple[
            Dict, Optional[Union[tf.keras.Model, tf.keras.Sequential]],
            Optional[TransformerEmbedding], Optional[Dict], Optional[Callable]
        ]:
    """
    MMD drift detector parameters to save.
    Note: only GaussianRBF kernel supported.

    Parameters
    ----------
    cd
        Drift detection object.
    """
    preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb = \
        preprocess_step_drift(cd._detector)
    if not isinstance(cd._detector.kernel, GaussianRBF):
        logger.warning('Currently only the default GaussianRBF kernel is supported.')
    sigma = cd._detector.kernel.sigma.numpy() if not cd._detector.infer_sigma else None
    state_dict = {
        'args':
            {
                'x_ref': cd._detector.x_ref,
            },
        'kwargs':
            {
                'p_val': cd._detector.p_val,
                'preprocess_x_ref': False,
                'update_x_ref': cd._detector.update_x_ref,
                'sigma': sigma,
                'configure_kernel_from_x_ref': not cd._detector.infer_sigma,
                'n_permutations': cd._detector.n_permutations,
                'input_shape': cd._detector.input_shape,
            },
        'other':
            {
                'n': cd._detector.n,
                'preprocess_x_ref': cd._detector.preprocess_x_ref,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
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
               filepath: Union[str, os.PathLike]) -> None:
    """
    Save TensorFlow components of OutlierAE

    Parameters
    ----------
    detector
        Outlier or adversarial detector object.
    filepath
        Save directory.
    """
    # create folder to save model in
    model_dir = Path(filepath).joinpath('model')
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)

    # save encoder, decoder and vae weights
    if isinstance(detector.ae.encoder.encoder_net, tf.keras.Sequential):
        detector.ae.encoder.encoder_net.save(model_dir.joinpath('encoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(detector.ae.decoder.decoder_net, tf.keras.Sequential):
        detector.ae.decoder.decoder_net.save(model_dir.joinpath('decoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(detector.ae, tf.keras.Model):
        detector.ae.save_weights(model_dir.joinpath('ae.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` ae detected. No ae saved.')


def save_tf_vae(detector: OutlierVAE,
                filepath: Union[str, os.PathLike]) -> None:
    """
    Save TensorFlow components of OutlierVAE.

    Parameters
    ----------
    detector
        Outlier detector object.
    filepath
        Save directory.
    """
    # create folder to save model in
    model_dir = Path(filepath).joinpath('model')
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)
    # save encoder, decoder and vae weights
    if isinstance(detector.vae.encoder.encoder_net, tf.keras.Sequential):
        detector.vae.encoder.encoder_net.save(model_dir.joinpath('encoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(detector.vae.decoder.decoder_net, tf.keras.Sequential):
        detector.vae.decoder.decoder_net.save(model_dir.joinpath('decoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(detector.vae, tf.keras.Model):
        detector.vae.save_weights(model_dir.joinpath('vae.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` vae detected. No vae saved.')


def save_tf_model(model: tf.keras.Model,
                  filepath: Union[str, os.PathLike],
                  save_dir: str = 'model',
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
        Name of folder to save to within the filepath directory.
    model_name
        Name of saved model.
    """
    # create folder to save model in
    model_dir = Path(filepath).joinpath(save_dir)
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)

    # save classification model
    if isinstance(model, tf.keras.Model) or isinstance(model, tf.keras.Sequential):
        model.save(model_dir.joinpath(model_name + '.h5'))
    else:
        logger.warning('No `tf.keras.Model` or `tf.keras.Sequential` detected. No model saved.')


def save_tf_llr(detector: LLR, filepath: Union[str, os.PathLike]) -> None:
    """
    Save LLR TensorFlow models or distributions.

    Parameters
    ----------
    detector
        Outlier detector object.
    filepath
        Save directory.
    """
    # create folder to save model in
    model_dir = Path(filepath).joinpath('model')
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)

    # Save LLR model
    if hasattr(detector, 'model_s') and hasattr(detector, 'model_b'):
        detector.model_s.save_weights(model_dir.joinpath('model_s.h5'))
        detector.model_b.save_weights(model_dir.joinpath('model_b.h5'))
    else:
        detector.dist_s.save(model_dir.joinpath('model.h5'))
        if detector.dist_b is not None:
            detector.dist_b.save(model_dir.joinpath('model_background.h5'))


def save_tf_hl(models: List[tf.keras.Model],
               filepath: Union[str, os.PathLike]) -> None:
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
        # create folder to save model in
        model_dir = Path(filepath).joinpath('model')
        if not model_dir.is_dir():
            logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
            model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        for i, m in enumerate(models):
            model_path = model_dir.joinpath('model_hl_' + str(i) + '.ckpt')
            m.save_weights(model_path)


def save_tf_aegmm(od: OutlierAEGMM,
                  filepath: Union[str, os.PathLike]) -> None:
    """
    Save TensorFlow components of OutlierAEGMM.

    Parameters
    ----------
    od
        Outlier detector object.
    filepath
        Save directory.
    """
    # create folder to save model in
    model_dir = Path(filepath).joinpath('model')
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)

    # save encoder, decoder, gmm density model and aegmm weights
    if isinstance(od.aegmm.encoder, tf.keras.Sequential):
        od.aegmm.encoder.save(model_dir.joinpath('encoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(od.aegmm.decoder, tf.keras.Sequential):
        od.aegmm.decoder.save(model_dir.joinpath('decoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(od.aegmm.gmm_density, tf.keras.Sequential):
        od.aegmm.gmm_density.save(model_dir.joinpath('gmm_density_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` GMM density net detected. No GMM density net saved.')
    if isinstance(od.aegmm, tf.keras.Model):
        od.aegmm.save_weights(model_dir.joinpath('aegmm.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` AEGMM detected. No AEGMM saved.')


def save_tf_vaegmm(od: OutlierVAEGMM,
                   filepath: Union[str, os.PathLike]) -> None:
    """
    Save TensorFlow components of OutlierVAEGMM.

    Parameters
    ----------
    od
        Outlier detector object.
    filepath
        Save directory.
    """
    # create folder to save model in
    model_dir = Path(filepath).joinpath('model')
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)

    # save encoder, decoder, gmm density model and vaegmm weights
    if isinstance(od.vaegmm.encoder.encoder_net, tf.keras.Sequential):
        od.vaegmm.encoder.encoder_net.save(model_dir.joinpath('encoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` encoder detected. No encoder saved.')
    if isinstance(od.vaegmm.decoder, tf.keras.Sequential):
        od.vaegmm.decoder.save(model_dir.joinpath('decoder_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` decoder detected. No decoder saved.')
    if isinstance(od.vaegmm.gmm_density, tf.keras.Sequential):
        od.vaegmm.gmm_density.save(model_dir.joinpath('gmm_density_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` GMM density net detected. No GMM density net saved.')
    if isinstance(od.vaegmm, tf.keras.Model):
        od.vaegmm.save_weights(model_dir.joinpath('vaegmm.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` VAEGMM detected. No VAEGMM saved.')


def save_tf_s2s(od: OutlierSeq2Seq,
                filepath: Union[str, os.PathLike]) -> None:
    """
    Save TensorFlow components of OutlierSeq2Seq.

    Parameters
    ----------
    od
        Outlier detector object.
    filepath
        Save directory.
    """
    # create folder to save model in
    model_dir = Path(filepath).joinpath('model')
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)

    # save seq2seq model weights and threshold estimation network
    if isinstance(od.seq2seq.threshold_net, tf.keras.Sequential):
        od.seq2seq.threshold_net.save(model_dir.joinpath('threshold_net.h5'))
    else:
        logger.warning('No `tf.keras.Sequential` threshold estimation net detected. No threshold net saved.')
    if isinstance(od.seq2seq, tf.keras.Model):
        od.seq2seq.save_weights(model_dir.joinpath('seq2seq.ckpt'))
    else:
        logger.warning('No `tf.keras.Model` Seq2Seq detected. No Seq2Seq model saved.')


def load_detector(filepath: Union[str, os.PathLike], **kwargs) -> Data:
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
    filepath = Path(filepath)
    if not filepath.is_dir():
        raise ValueError('{} does not exist.'.format(filepath))

    # Check if dill files exist, otherwise check for pickle files, otherwise raise error
    files = [str(f.name) for f in filepath.iterdir() if f.is_file()]
    if 'meta.dill' in files:
        suffix = '.dill'
    elif 'meta.pickle' in files:
        suffix = '.pickle'
    else:
        raise ValueError('Neither meta.dill or meta.pickle exist in {}.'.format(filepath))

    # load metadata
    meta_dict = dill.load(open(filepath.joinpath('meta' + suffix), 'rb'))

    # check version
    try:
        if meta_dict['version'] != __version__:
            warnings.warn(f'Trying to load detector from version {meta_dict["version"]} when using version '
                          f'{__version__}. This may lead to breaking code or invalid results.')
    except KeyError:
        warnings.warn('Trying to load detector from an older version.'
                      'This may lead to breaking code or invalid results.')

    if 'backend' in list(meta_dict.keys()) and meta_dict['backend'] == 'pytorch':
        raise NotImplementedError('Detectors with PyTorch backend are not yet supported.')

    detector_name = meta_dict['name']
    if detector_name not in DEFAULT_DETECTORS:
        raise ValueError('{} is not supported by `load_detector`.'.format(detector_name))

    # load outlier detector specific parameters
    state_dict = dill.load(open(filepath.joinpath(detector_name + suffix), 'rb'))

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
    elif detector_name in ['ChiSquareDrift', 'ClassifierDriftTF', 'KSDrift', 'MMDDriftTF', 'TabularDrift']:
        emb, tokenizer = None, None
        if state_dict['other']['load_text_embedding']:
            emb, tokenizer = load_text_embed(filepath)
        model = load_tf_model(filepath, model_name='encoder')
        if detector_name == 'KSDrift':
            load_fn = init_cd_ksdrift
        elif detector_name == 'MMDDriftTF':
            load_fn = init_cd_mmddrift
        elif detector_name == 'ChiSquareDrift':
            load_fn = init_cd_chisquaredrift
        elif detector_name == 'TabularDrift':
            load_fn = init_cd_tabulardrift
        elif detector_name == 'ClassifierDriftTF':
            clf_drift = load_tf_model(filepath, model_name='clf_drift')
            load_fn = partial(init_cd_classifierdrift, clf_drift)
        else:
            raise NotImplementedError
        detector = load_fn(state_dict, model, emb, tokenizer, **kwargs)
    elif detector_name == 'LLR':
        models = load_tf_llr(filepath, **kwargs)
        detector = init_od_llr(state_dict, models)

    detector.meta = meta_dict
    return detector


def load_tf_model(filepath: Union[str, os.PathLike],
                  load_dir: str = 'model',
                  custom_objects: dict = None,
                  model_name: str = 'model') -> tf.keras.Model:
    """
    Load TensorFlow model.

    Parameters
    ----------
    filepath
        Saved model directory.
    load_dir
            Name of saved model folder within the filepath directory.
    custom_objects
        Optional custom objects when loading the TensorFlow model.
    model_name
        Name of loaded model.

    Returns
    -------
    Loaded model.
    """
    model_dir = Path(filepath).joinpath(load_dir)
    # Check if path exists
    if not model_dir.is_dir():
        logger.warning('Directory {} does not exist.'.format(model_dir))
        return None
    # Check if model exists
    if model_name + '.h5' not in [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No model found in {}.'.format(model_dir))
        return None
    model = tf.keras.models.load_model(model_dir.joinpath(model_name + '.h5'), custom_objects=custom_objects)
    return model


def load_tf_hl(filepath: Union[str, os.PathLike], model: tf.keras.Model, state_dict: dict) -> List[tf.keras.Model]:
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
    model_dir = Path(filepath).joinpath('model')
    hidden_layer_kld = state_dict['hidden_layer_kld']
    if not hidden_layer_kld:
        return []
    model_hl = []
    for i, (hidden_layer, output_dim) in enumerate(hidden_layer_kld.items()):
        m = DenseHidden(model, hidden_layer, output_dim)
        m.load_weights(model_dir.joinpath('model_hl_' + str(i) + '.ckpt'))
        model_hl.append(m)
    return model_hl


def load_tf_ae(filepath: Union[str, os.PathLike]) -> tf.keras.Model:
    """
    Load AE.

    Parameters
    ----------
    filepath
        Saved model directory.

    Returns
    -------
    Loaded AE.
    """
    model_dir = Path(filepath).joinpath('model')
    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No encoder, decoder or ae found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir.joinpath('encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(model_dir.joinpath('decoder_net.h5'))
    ae = AE(encoder_net, decoder_net)
    ae.load_weights(model_dir.joinpath('ae.ckpt'))
    return ae


def load_tf_vae(filepath: Union[str, os.PathLike],
                state_dict: Dict) -> tf.keras.Model:
    """
    Load VAE.

    Parameters
    ----------
    filepath
        Saved model directory.
    state_dict
        Dictionary containing the latent dimension and beta parameters.

    Returns
    -------
    Loaded VAE.
    """
    model_dir = Path(filepath).joinpath('model')
    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No encoder, decoder or vae found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir.joinpath('encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(model_dir.joinpath('decoder_net.h5'))
    vae = VAE(encoder_net, decoder_net, state_dict['latent_dim'], beta=state_dict['beta'])
    vae.load_weights(model_dir.joinpath('vae.ckpt'))
    return vae


def load_tf_aegmm(filepath: Union[str, os.PathLike],
                  state_dict: Dict) -> tf.keras.Model:
    """
    Load AEGMM.

    Parameters
    ----------
    filepath
        Saved model directory.
    state_dict
        Dictionary containing the `n_gmm` and `recon_features` parameters.

    Returns
    -------
    Loaded AEGMM.
    """
    model_dir = Path(filepath).joinpath('model')

    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No encoder, decoder, gmm density net or aegmm found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir.joinpath('encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(model_dir.joinpath('decoder_net.h5'))
    gmm_density_net = tf.keras.models.load_model(model_dir.joinpath('gmm_density_net.h5'))
    aegmm = AEGMM(encoder_net, decoder_net, gmm_density_net, state_dict['n_gmm'], state_dict['recon_features'])
    aegmm.load_weights(model_dir.joinpath('aegmm.ckpt'))
    return aegmm


def load_tf_vaegmm(filepath: Union[str, os.PathLike],
                   state_dict: Dict) -> tf.keras.Model:
    """
    Load VAEGMM.

    Parameters
    ----------
    filepath
        Saved model directory.
    state_dict
        Dictionary containing the `n_gmm`, `latent_dim` and `recon_features` parameters.

    Returns
    -------
    Loaded VAEGMM.
    """
    model_dir = Path(filepath).joinpath('model')
    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No encoder, decoder, gmm density net or vaegmm found in {}.'.format(model_dir))
        return None
    encoder_net = tf.keras.models.load_model(model_dir.joinpath('encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(model_dir.joinpath('decoder_net.h5'))
    gmm_density_net = tf.keras.models.load_model(model_dir.joinpath('gmm_density_net.h5'))
    vaegmm = VAEGMM(encoder_net, decoder_net, gmm_density_net, state_dict['n_gmm'],
                    state_dict['latent_dim'], state_dict['recon_features'], state_dict['beta'])
    vaegmm.load_weights(model_dir.joinpath('vaegmm.ckpt'))
    return vaegmm


def load_tf_s2s(filepath: Union[str, os.PathLike],
                state_dict: Dict) -> tf.keras.Model:
    """
    Load seq2seq TensorFlow model.

    Parameters
    ----------
    filepath
        Saved model directory.
    state_dict
        Dictionary containing the `latent_dim`, `shape`, `output_activation` and `beta` parameters.

    Returns
    -------
    Loaded seq2seq model.
    """
    model_dir = Path(filepath).joinpath('model')
    if not [f.name for f in model_dir.glob('[!.]*.h5')]:
        logger.warning('No seq2seq or threshold estimation net found in {}.'.format(model_dir))
        return None
    # load threshold estimator net, initialize encoder and decoder and load seq2seq weights
    threshold_net = tf.keras.models.load_model(model_dir.joinpath('threshold_net.h5'), compile=False)
    latent_dim = state_dict['latent_dim']
    n_features = state_dict['shape'][-1]
    encoder_net = EncoderLSTM(latent_dim)
    decoder_net = DecoderLSTM(latent_dim, n_features, state_dict['output_activation'])
    seq2seq = Seq2Seq(encoder_net, decoder_net, threshold_net, n_features, beta=state_dict['beta'])
    seq2seq.load_weights(model_dir.joinpath('seq2seq.ckpt'))
    return seq2seq


def load_tf_llr(filepath: Union[str, os.PathLike], dist_s: Union[Distribution, PixelCNN] = None,
                dist_b: Union[Distribution, PixelCNN] = None, input_shape: tuple = None):
    """
    Load LLR TensorFlow models or distributions.

    Parameters
    ----------
    detector
        Likelihood ratio detector.
    filepath
        Saved model directory.
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
    model_dir = Path(filepath).joinpath('model')
    h5files = [f.name for f in model_dir.glob('[!.]*.h5')]
    if 'model_s.h5' in h5files and 'model_b.h5' in h5files:
        model_s, dist_s = build_model(dist_s, input_shape, str(model_dir.joinpath('model_s.h5').resolve()))
        model_b, dist_b = build_model(dist_b, input_shape, str(model_dir.joinpath('model_b.h5').resolve()))
        return dist_s, dist_b, model_s, model_b
    else:
        dist_s = tf.keras.models.load_model(model_dir.joinpath('model.h5'), compile=False)
        if 'model_background.h5' in h5files:
            dist_b = tf.keras.models.load_model(model_dir.joinpath('model_background.h5'), compile=False)
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


def load_text_embed(filepath: Union[str, os.PathLike], load_dir: str = 'model') \
        -> Tuple[TransformerEmbedding, Callable]:
    model_dir = Path(filepath).joinpath(load_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()))
    args = dill.load(open(model_dir.joinpath('embedding.dill'), 'rb'))
    emb = TransformerEmbedding(
        str(model_dir.resolve()), embedding_type=args['embedding_type'], layers=args['layers']
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
            model = _Encoder(emb, mlp=model)
            preprocess_kwargs['tokenizer'] = tokenizer
        preprocess_kwargs['model'] = UAE(encoder_net=model)
    else:  # incl. preprocess_kwargs['model'] == 'HiddenOutput'
        preprocess_kwargs['model'] = model

    return preprocess_fn, preprocess_kwargs


def init_cd_classifierdrift(clf_drift: tf.keras.Model, state_dict: Dict, model: Optional[tf.keras.Model],
                            emb: Optional[TransformerEmbedding], tokenizer: Optional[Callable], **kwargs) \
        -> ClassifierDrift:
    """
    Initialize ClassifierDrift detector.

    Parameters
    ----------
    clf_drift
        Model used for drift classification.
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
    Initialized ClassifierDrift instance.
    """
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict['other'], model, emb, tokenizer, **kwargs)
    if isinstance(preprocess_fn, Callable) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    state_dict['kwargs']['train_kwargs']['optimizer'] = \
        tf.keras.optimizers.get(state_dict['kwargs']['train_kwargs']['optimizer'])
    args = list(state_dict['args'].values()) + [clf_drift]
    cd = ClassifierDrift(*args, **state_dict['kwargs'])
    attrs = state_dict['other']
    cd._detector.n = attrs['n']
    cd._detector.preprocess_x_ref = attrs['preprocess_x_ref']
    cd._detector.skf = attrs['skf']
    return cd


def init_cd_chisquaredrift(state_dict: Dict, model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
                           emb: Optional[TransformerEmbedding], tokenizer: Optional[Callable], **kwargs) \
        -> ChiSquareDrift:
    """
    Initialize ChiSquareDrift detector.

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
    Initialized ChiSquareDrift instance.
    """
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict['other'], model, emb, tokenizer, **kwargs)
    if isinstance(preprocess_fn, Callable) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = ChiSquareDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    attrs = state_dict['other']
    cd.n = attrs['n']
    cd.preprocess_x_ref = attrs['preprocess_x_ref']
    return cd


def init_cd_tabulardrift(state_dict: Dict, model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
                         emb: Optional[TransformerEmbedding], tokenizer: Optional[Callable], **kwargs) \
        -> TabularDrift:
    """
    Initialize TabularDrift detector.

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
    Initialized TabularDrift instance.
    """
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict['other'], model, emb, tokenizer, **kwargs)
    if isinstance(preprocess_fn, Callable) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = TabularDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    attrs = state_dict['other']
    cd.n = attrs['n']
    cd.preprocess_x_ref = attrs['preprocess_x_ref']
    return cd


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
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict['other'], model, emb, tokenizer, **kwargs)
    if isinstance(preprocess_fn, Callable) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = KSDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    attrs = state_dict['other']
    cd.n = attrs['n']
    cd.preprocess_x_ref = attrs['preprocess_x_ref']
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
    preprocess_fn, preprocess_kwargs = init_preprocess(state_dict['other'], model, emb, tokenizer, **kwargs)
    if isinstance(preprocess_fn, Callable) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = MMDDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    attrs = state_dict['other']
    cd._detector.n = attrs['n']
    cd._detector.preprocess_x_ref = attrs['preprocess_x_ref']
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
