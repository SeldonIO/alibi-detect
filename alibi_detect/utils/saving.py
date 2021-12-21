# type: ignore
# TODO: need to rewrite utilities using isinstance or @singledispatch for type checking to work properly
# TODO: Need to clarify public vs private functions here
# TODO: Add verbose functionality
# TODO: clean up save directories, filenames etc, fix save_tf_model directory logic (see #348)
import dill
import toml
import numpy as np
from functools import partial
import logging
import os
from pathlib import Path
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer
from tensorflow_probability.python.distributions.distribution import Distribution
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal
from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.ad.adversarialae import DenseHidden
from alibi_detect.cd import ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, LSDDDrift, TabularDrift, \
    CVMDrift, FETDrift
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.cd.tensorflow.preprocess import _Encoder
from alibi_detect.models.tensorflow.autoencoder import AE, AEGMM, DecoderLSTM, EncoderLSTM, Seq2Seq, VAE, VAEGMM
from alibi_detect.models.tensorflow import PixelCNN, TransformerEmbedding
from alibi_detect.od import (IForest, LLR, Mahalanobis, OutlierAE, OutlierAEGMM, OutlierProphet,
                             OutlierSeq2Seq, OutlierVAE, OutlierVAEGMM, SpectralResidual)
from alibi_detect.od.llr import build_model
from alibi_detect.utils.loading import _load_detector_config, load_tf_model, SUPPORTED_MODELS, SupportedModels
from alibi_detect.utils.registry import registry
from alibi_detect.version import __version__

# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

logger = logging.getLogger(__name__)

Data = Union[
    AdversarialAE,
    ChiSquareDrift,
    ClassifierDrift,
    IForest,
    KSDrift,
    LLR,
    Mahalanobis,
    MMDDrift,
    LSDDDrift,
    ModelDistillation,
    OutlierAE,
    OutlierAEGMM,
    OutlierProphet,
    OutlierSeq2Seq,
    OutlierVAE,
    OutlierVAEGMM,
    SpectralResidual,
    TabularDrift,
    CVMDrift,
    FETDrift
]

DEFAULT_DETECTORS = [
    'AdversarialAE',
    'IForest',
    'LLR',
    'Mahalanobis',
    'ModelDistillation',
    'OutlierAE',
    'OutlierAEGMM',
    'OutlierProphet',
    'OutlierSeq2Seq',
    'OutlierVAE',
    'OutlierVAEGMM',
    'SpectralResidual',
]

# TODO - add all drift methods in as .get_config() methods are completed
DRIFT_DETECTORS = [  # Drift detectors separated out as they now have their own save methods
    'MMDDrift',
    'LSDDDrift',
    'ChiSquareDrift',
    'TabularDrift',
    'KSDrift',
    'CVMDrift',
    'FETDrift',
    'ClassifierDrift',
]


def save_detector(detector: Data, filepath: Union[str, os.PathLike], verbose: bool = False) -> None:
    """
    Save outlier, drift or adversarial detector.

    Parameters
    ----------
    detector
        Detector object.
    filepath
        Save directory.
    verbose
        Whether to print progress messages.
    """
    if 'backend' in list(detector.meta.keys()) and detector.meta['backend'] == 'pytorch':
        raise NotImplementedError('Detectors with PyTorch backend are not yet supported.')

    detector_name = detector.__class__.__name__
    if detector_name not in DEFAULT_DETECTORS and detector_name not in DRIFT_DETECTORS:
        raise ValueError('{} is not supported by `save_detector`.'.format(detector_name))

    # check if path exists
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # If a drift detector, wrap drift detector save method
    if detector_name in DRIFT_DETECTORS:
        _save_detector_config(detector, filepath, verbose=verbose)

    # Otherwise, save via the previous meta and state_dict approach
    else:
        # save metadata
        with open(filepath.joinpath('meta.dill'), 'wb') as f:
            dill.dump(detector.meta, f)

        # save outlier detector specific parameters
        if detector_name == 'OutlierAE':
            state_dict = _state_ae(detector)
        elif detector_name == 'OutlierVAE':
            state_dict = _state_vae(detector)
        elif detector_name == 'Mahalanobis':
            state_dict = _state_mahalanobis(detector)
        elif detector_name == 'IForest':
            state_dict = _state_iforest(detector)
        elif detector_name == 'OutlierAEGMM':
            state_dict = _state_aegmm(detector)
        elif detector_name == 'OutlierVAEGMM':
            state_dict = _state_vaegmm(detector)
        elif detector_name == 'AdversarialAE':
            state_dict = _state_adv_ae(detector)
        elif detector_name == 'ModelDistillation':
            state_dict = _state_adv_md(detector)
        elif detector_name == 'OutlierProphet':
            state_dict = _state_prophet(detector)
        elif detector_name == 'SpectralResidual':
            state_dict = _state_sr(detector)
        elif detector_name == 'OutlierSeq2Seq':
            state_dict = _state_s2s(detector)
        elif detector_name == 'LLR':
            state_dict = _state_llr(detector)

        with open(filepath.joinpath(detector_name + '.dill'), 'wb') as f:
            dill.dump(state_dict, f)

        # save outlier detector specific TensorFlow models
        if detector_name == 'OutlierAE':
            save_tf_ae(detector, filepath)
        elif detector_name == 'OutlierVAE':
            save_tf_vae(detector, filepath)
        elif detector_name == 'OutlierAEGMM':
            save_tf_aegmm(detector, filepath)
        elif detector_name == 'OutlierVAEGMM':
            save_tf_vaegmm(detector, filepath)
        elif detector_name == 'AdversarialAE':
            save_tf_ae(detector, filepath)
            save_tf_model(detector.model, filepath)
            save_tf_hl(detector.model_hl, filepath)
        elif detector_name == 'ModelDistillation':
            save_tf_model(detector.distilled_model, filepath, save_dir='distilled_model')
            save_tf_model(detector.model, filepath, save_dir='model')
        elif detector_name == 'OutlierSeq2Seq':
            save_tf_s2s(detector, filepath)
        elif detector_name == 'LLR':
            save_tf_llr(detector, filepath)


# TODO - eventually this will become save_detector
def _save_detector_config(detector: Data, filepath: Union[str, os.PathLike], verbose: bool = False):
    """
    Save a drift detector. The detector is saved as a yaml config file. Artefacts such as
    `preprocess_fn`, models, embeddings, tokenizers etc are serialized, and their filepaths are
    added to the config file.

    The detector can be loaded again by passing the resulting config file or filepath to `load_detector`.

    Parameters
    ----------
    detector
        The detector to save.
    filepath
        File path to save serialized artefacts to.
    verbose
        Whether to print progress messages.
    """
    # Get backend, input_shape and detector_name
    backend = detector.meta.get('backend', 'tensorflow')
    # TODO - setting to tensorflow by default atm, but what do we do about detectors with no backend. (We still need to
    #  know whether preprocess_fn artefacts are tensorflow or pytorch.
    if backend == 'pytorch':
        raise NotImplementedError('Detectors with PyTorch backend are not yet supported.')
    detector_name = detector.__class__.__name__
    if detector_name not in DRIFT_DETECTORS:
        raise ValueError('{} is not supported by `save_drift_detector`.'.format(detector_name))

    # Process file paths
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # Get the detector config (with artefacts still within it)
    cfg = detector.get_config()

    # Save x_ref
    save_path = filepath.joinpath('x_ref.npy')
    np.save(str(save_path), cfg['x_ref'])
    cfg.update({'x_ref': 'x_ref.npy'})

    # Save preprocess_fn
    preprocess_fn = cfg.get('preprocess_fn', None)
    if preprocess_fn is not None:
        preprocess_cfg = serialize_preprocess(preprocess_fn, backend, cfg['input_shape'], filepath, verbose)
        cfg['preprocess_fn'] = preprocess_cfg

    # Serialize kernel
    kernel = cfg.get('kernel', None)
    if kernel is not None:
        device = detector.device.type if hasattr(detector, 'device') else None
        kernel_cfg = save_kernel(kernel, filepath, device, verbose)
        cfg['kernel'] = kernel_cfg

    # ClassifierDrift and SpotTheDiffDrift specific artefacts.
    # Serialize detector model
    model = cfg.get('model', None)
    if model is not None:
        model_cfg, _ = save_model(model, base_path=filepath, input_shape=cfg['input_shape'],
                                  backend=backend, verbose=verbose)
        cfg['model'] = model_cfg

    # Serialize dataset
    dataset = cfg.get('dataset', None)
    if dataset is not None:
        dataset_cfg, dataset_kwargs = serialize_function(dataset, filepath, Path('dataset'))
        cfg.update({'dataset': dataset_cfg})
        if len(dataset_kwargs) != 0:
            cfg['dataset']['kwargs'] = dataset_kwargs

    # Serialize reg_loss_fn
    reg_loss_fn = cfg.get('reg_loss_fn', None)
    if reg_loss_fn is not None:
        reg_loss_fn_cfg, _ = serialize_function(reg_loss_fn, filepath, Path('reg_loss_fn'))
        cfg['reg_loss_fn'] = reg_loss_fn_cfg

    # Save initial_diffs
    initial_diffs = cfg.get('initial_diffs', None)
    if initial_diffs is not None:
        save_path = filepath.joinpath('initial_diffs.npy')
        np.save(str(save_path), initial_diffs)
        cfg.update({'initial_diffs': 'initial_diffs.npy'})

    # Save config
    cfg = _resolve_paths(cfg)
    with open(filepath.joinpath('config.toml'), 'w') as f:
        toml.dump(cfg, f, encoder=toml.TomlNumpyEncoder())


def _state_iforest(od: IForest) -> Dict:
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


def _state_mahalanobis(od: Mahalanobis) -> Dict:
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


def _state_ae(od: OutlierAE) -> Dict:
    """
    OutlierAE parameters to save.

    Parameters
    ----------
    od
        Outlier detector object.
    """
    state_dict = {'threshold': od.threshold}
    return state_dict


def _state_vae(od: OutlierVAE) -> Dict:
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


def _state_aegmm(od: OutlierAEGMM) -> Dict:
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


def _state_vaegmm(od: OutlierVAEGMM) -> Dict:
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


def _state_adv_ae(ad: AdversarialAE) -> Dict:
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


def _state_adv_md(md: ModelDistillation) -> Dict:
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


def _state_prophet(od: OutlierProphet) -> Dict:
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


def _state_sr(od: SpectralResidual) -> Dict:
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


def _state_s2s(od: OutlierSeq2Seq) -> Dict:
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


def _state_llr(od: LLR) -> Dict:
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
                  save_dir: Union[str, os.PathLike] = 'model',
                  save_format: Literal['tf', 'h5'] = 'h5') -> None:  # TODO - change to tf, later PR?
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
    save_format
        The format to save to. 'tf' to save to the newer SavedModel format, 'h5' to save to the lighter-weight
        legacy hdf5 format.
    """
    # create folder to save model in
    model_path = Path(filepath).joinpath(save_dir)
    if not model_path.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_path))
        model_path.mkdir(parents=True, exist_ok=True)

    # save classification model
    model_path = model_path.joinpath('model.h5') if save_format == 'h5' else model_path

    if isinstance(model, tf.keras.Model) or isinstance(model, tf.keras.Sequential):
        model.save(model_path, save_format=save_format)
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
    filepath = Path(filepath)
    # If reference is a 'config.toml' itself, pass to new load function
    if filepath.name == 'config.toml':
        return _load_detector_config(filepath)

    # Otherwise, if a directory, look for meta.dill, meta.pickle or config.toml inside it
    elif filepath.is_dir():
        files = [str(f.name) for f in filepath.iterdir() if f.is_file()]
        if 'config.toml' in files:
            return _load_detector_config(filepath.joinpath('config.toml'))
        elif 'meta.dill' in files:
            return _load_detector_legacy(filepath, '.dill', **kwargs)
        elif 'meta.pickle' in files:
            return _load_detector_legacy(filepath, '.pickle', **kwargs)
        else:
            raise ValueError('Neither meta.dill, meta.pickle or config.toml exist in {}.'.format(filepath))

    # No other file types are accepted, so if not dir raise error
    else:
        raise ValueError("load_detector accepts only a filepath to a directory, or a config.toml file.")


def _load_detector_legacy(filepath: Union[str, os.PathLike], suffix: str, **kwargs) -> Data:
    """
    Legacy function to load outlier, drift or adversarial detectors stored dill or pickle files.

    Warning
    -------
    This function will be removed in a future version.

    Parameters
    ----------
    filepath
        Load directory.
    suffix
        File suffix for meta and state files. Either `'.dill'` or `'.pickle'`.

    Returns
    -------
    Loaded outlier or adversarial detector object.
    """
    warnings.warn(' Loading of meta.dill and meta.pickle files will be removed in a future version.',
                  DeprecationWarning, 3)

    if kwargs:
        k = list(kwargs.keys())
    else:
        k = []

    # check if path exists
    filepath = Path(filepath)
    if not filepath.is_dir():
        raise ValueError('{} does not exist.'.format(filepath))

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
    if detector_name not in DEFAULT_DETECTORS and detector_name not in DRIFT_DETECTORS:
        raise ValueError('{} is not supported by `load_detector`.'.format(detector_name))

    # load outlier detector specific parameters
    state_dict = dill.load(open(filepath.joinpath(detector_name + suffix), 'rb'))

    # initialize outlier detector
    if detector_name == 'OutlierAE':
        ae = load_tf_ae(filepath)
        detector = _init_od_ae(state_dict, ae)
    elif detector_name == 'OutlierVAE':
        vae = load_tf_vae(filepath, state_dict)
        detector = _init_od_vae(state_dict, vae)
    elif detector_name == 'Mahalanobis':
        detector = _init_od_mahalanobis(state_dict)
    elif detector_name == 'IForest':
        detector = _init_od_iforest(state_dict)
    elif detector_name == 'OutlierAEGMM':
        aegmm = load_tf_aegmm(filepath, state_dict)
        detector = _init_od_aegmm(state_dict, aegmm)
    elif detector_name == 'OutlierVAEGMM':
        vaegmm = load_tf_vaegmm(filepath, state_dict)
        detector = _init_od_vaegmm(state_dict, vaegmm)
    elif detector_name == 'AdversarialAE':
        ae = load_tf_ae(filepath)
        custom_objects = kwargs['custom_objects'] if 'custom_objects' in k else None
        model = load_tf_model(filepath, custom_objects=custom_objects)
        model_hl = load_tf_hl(filepath, model, state_dict)
        detector = _init_ad_ae(state_dict, ae, model, model_hl)
    elif detector_name == 'ModelDistillation':
        md = load_tf_model(filepath, model_name='distilled_model')
        custom_objects = kwargs['custom_objects'] if 'custom_objects' in k else None
        model = load_tf_model(filepath, custom_objects=custom_objects)
        detector = _init_ad_md(state_dict, md, model)
    elif detector_name == 'OutlierProphet':
        detector = _init_od_prophet(state_dict)
    elif detector_name == 'SpectralResidual':
        detector = _init_od_sr(state_dict)
    elif detector_name == 'OutlierSeq2Seq':
        seq2seq = load_tf_s2s(filepath, state_dict)
        detector = _init_od_s2s(state_dict, seq2seq)
    elif detector_name == 'LLR':
        models = load_tf_llr(filepath, **kwargs)
        detector = _init_od_llr(state_dict, models)
    else:
        raise NotImplementedError

    detector.meta = meta_dict
    return detector


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


def _init_od_ae(state_dict: Dict,
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


def _init_od_vae(state_dict: Dict,
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


def _init_ad_ae(state_dict: Dict,
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


def _init_ad_md(state_dict: Dict,
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


def _init_od_aegmm(state_dict: Dict,
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


def _init_od_vaegmm(state_dict: Dict,
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


def _init_od_s2s(state_dict: Dict,
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


def _load_text_embed(filepath: Union[str, os.PathLike], load_dir: str = 'model') \
        -> Tuple[TransformerEmbedding, Callable]:
    model_dir = Path(filepath).joinpath(load_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()))
    args = dill.load(open(model_dir.joinpath('embedding.dill'), 'rb'))
    emb = TransformerEmbedding(
        str(model_dir.resolve()), embedding_type=args['embedding_type'], layers=args['layers']
    )
    return emb, tokenizer


def _init_preprocess(state_dict: Dict, model: Optional[Union[tf.keras.Model, tf.keras.Sequential]],
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


def _init_od_mahalanobis(state_dict: Dict) -> Mahalanobis:
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


def _init_od_iforest(state_dict: Dict) -> IForest:
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


def _init_od_prophet(state_dict: Dict) -> OutlierProphet:
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


def _init_od_sr(state_dict: Dict) -> SpectralResidual:
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


def _init_od_llr(state_dict: Dict, models: tuple) -> LLR:
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


def serialize_preprocess(preprocess_fn: Callable,
                         backend: str,
                         input_shape: Optional[tuple],
                         filepath: Path,
                         verbose: bool = False) -> dict:
    """
    Serializes a drift detectors preprocess_fn. Artefacts are saved to disk, and a config dict containing filepaths
    to the saved artefacts is returned.

    Parameters
    ----------
    preprocess_fn
        The preprocess function to be serialized.
    backend
        Specifies the detectors backend (if it has one). Either `'tensorflow'`, `'pytorch'` or `None`.
    input_shape
        Input shape for a model (if a model exists).
    filepath
        Directory to save serialized artefacts to.
    verbose
        Verbose logging.

    Returns
    -------
    The config dictionary, containing references to the serialized artefacts. The format if this dict matches that
    of the `preprocess` field in the drift detector specification.
    """
    preprocess_cfg = {}
    local_path = Path('preprocess_fn')

    # Serialize function
    func, func_kwargs = serialize_function(preprocess_fn, filepath, local_path.joinpath('function'))
    preprocess_cfg.update({'src': func})

    # Process partial function kwargs (if they exist)
    kwargs = {}
    for k, v in func_kwargs.items():
        # Model/embedding
        if isinstance(v, SupportedModels):
            cfg_model, cfg_embed = save_model(v, filepath, input_shape, backend, local_path, verbose)
            kwargs.update({k: cfg_model})
            if cfg_embed is not None:
                kwargs.update({'embedding': cfg_embed})

        # Tokenizer
        elif isinstance(v, PreTrainedTokenizerBase):
            cfg_token = save_tokenizer(v, filepath, local_path, verbose)
            kwargs.update({k: cfg_token})

        # Arbitrary function
        elif callable(v):
            with open(filepath.joinpath(k + '.dill'), 'wb') as f:
                dill.dump(v, f)
            kwargs.update({k: local_path.joinpath(k + '.dill')})

        # Put remaining kwargs directly into cfg
        else:
            kwargs.update({k: v})

    if 'preprocess_drift' in func:
        preprocess_cfg.update(kwargs)
    else:
        kwargs.update({'kwargs': kwargs})

    return preprocess_cfg


def serialize_function(func: Callable, base_path: Path, local_path: Path = Path('function')) -> Tuple[str, dict]:

    # If a partial, save function and kwargs
    if isinstance(func, partial):
        kwargs = func.keywords
        func = func.func
    else:
        kwargs = {}

    # If a registered function, save registry string
    keys = [k for k, v in registry.get_all().items() if func == v]
    registry_str = keys[0] if len(keys) == 1 else None
    if registry_str is not None:  # alibi-detect registered function
        src = '@' + registry_str

    # Otherwise, save as dill
    else:
        # create folder to save model in
        filepath = base_path.joinpath(local_path)
        if not filepath.parent.is_dir():
            logger.warning('Directory {} does not exist and is now created.'.format(filepath.parent))
            filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath.with_suffix('.dill'), 'wb') as f:
            dill.dump(func, f)
        src = local_path.with_suffix('.dill')

    return src, kwargs


def save_embedding(embed: tf.keras.Model,
                   embed_args: dict,
                   filepath: Path) -> None:
    """
    Save embeddings for text drift models.

    Parameters
    ----------
    embed
        Embedding model.
    embed_args
        Arguments for TransformerEmbedding module.
    filepath
        The save directory.
    """
    # create folder to save model in
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # Save embedding model
    embed.save_pretrained(filepath)
    with open(filepath.joinpath('embedding.dill'), 'wb') as f:
        dill.dump(embed_args, f)


def _resolve_paths(cfg: dict, absolute: bool = False) -> dict:
    for k, v in cfg.items():
        if isinstance(v, dict):
            _resolve_paths(v, absolute)
        elif isinstance(v, Path):
            if absolute:
                v = v.resolve()
            cfg.update({k: str(v)})
    return cfg


def save_model(model: SUPPORTED_MODELS,
               base_path: Path,
               input_shape: tuple,
               backend: str,
               path: Path = Path('.'),
               verbose: bool = False) -> Tuple[dict, Optional[dict]]:
    filepath = base_path.joinpath(path)
    # create folder to save model in
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    if backend == 'tensorflow':
        cfg_model, cfg_embed = {}, None
        if isinstance(model, UAE):
            if isinstance(model.encoder.layers[0], TransformerEmbedding):  # text drift
                # embedding
                cfg_embed = {}
                embed = model.encoder.layers[0].model
                cfg_embed.update({'type': model.encoder.layers[0].emb_type})
                cfg_embed.update({'layers': model.encoder.layers[0].hs_emb.keywords['layers']})
                save_embedding(embed, cfg_embed, filepath.joinpath('embedding'))
                cfg_embed.update({'src': path.joinpath('embedding')})
                # preprocessing encoder
                inputs = Input(shape=input_shape, dtype=tf.int64)
                model.encoder.call(inputs)
                shape_enc = (model.encoder.layers[0].output.shape[-1],)
                layers = [InputLayer(input_shape=shape_enc)] + model.encoder.layers[1:]
                model = tf.keras.Sequential(layers)
                _ = model(tf.zeros((1,) + shape_enc))
            else:
                model = model.encoder
            cfg_model.update({'type': 'UAE'})

        elif isinstance(model, HiddenOutput):
            model = model.model
            cfg_model.update({'type': 'HiddenOutput'})
        elif isinstance(model, (tf.keras.Sequential, tf.keras.Model)):
            model = model
            cfg_model.update({'type': 'custom'})

        save_tf_model(model, filepath=filepath, save_dir=path.joinpath('model'))

    else:
        raise NotImplementedError("Saving of pytorch models is not yet implemented.")

    cfg_model.update({'src': path.joinpath('model')})
    return cfg_model, cfg_embed


def save_tokenizer(tokenizer: PreTrainedTokenizerBase,
                   base_path: Path,
                   path: Path = Path('.'),
                   verbose: bool = False) -> dict:
    # create folder to save model in
    filepath = base_path.joinpath(path)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    cfg_token = {}
    tokenizer.save_pretrained(filepath.joinpath('tokenizer'))
    cfg_token.update({'src': path.joinpath('tokenizer')})
    return cfg_token


def save_kernel(kernel: Callable,
                filepath: Path,
                device: Optional[str],
                verbose: bool = False) -> dict:
    """
    Function to save kernel. If the kernel is stored in the artefact registry, the registry key (and kwargs) are
    written to config. If the kernel is a generic callable, it is pickled.

    Parameters
    ----------
    kernel
        The kernel to save.
    filepath
        Filepath to save to (if the kernel is a generic callable).
    device
        Device. Only needed if pytorch backend being used.
    verbose
        Whether to print progress messages.

    Returns
    -------
    The kernel config dictionary.
    """
    cfg_kernel = {}

    keys = [k for k, v in registry.get_all().items() if isinstance(kernel, v)]
    registry_str = keys[0] if len(keys) == 1 else None
    if registry_str is not None:  # alibi-detect registered kernel
        cfg_kernel.update({'src': '@' + registry_str})

        # kwargs for registered kernel - #NOTE: Potentially would need updating if new kernels registered
        sigma = kernel.sigma if hasattr(kernel, 'sigma') else None
        sigma = sigma.cpu() if device == 'cuda' else sigma
        cfg_kernel.update({
            'sigma': sigma.numpy().tolist(),
            'trainable': kernel.trainable if hasattr(kernel, 'trainable') else None
        })

    elif isinstance(kernel, Callable):  # generic callable
        with open(filepath.joinpath('kernel.dill'), 'wb') as f:
            dill.dump(kernel, f)
        cfg_kernel.update({'src': 'kernel.dill'})

    else:  # kernel could not be saved
        raise ValueError("Could not save kernel. Is it a valid Callable or a alibi-detect registered kernel?")

    return cfg_kernel
