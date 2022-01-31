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
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer
from transformers import PreTrainedTokenizerBase
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal
from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.od import (IForest, LLR, Mahalanobis, OutlierAE, OutlierAEGMM, OutlierProphet,
                             OutlierSeq2Seq, OutlierVAE, OutlierVAEGMM, SpectralResidual)
from alibi_detect.utils.loading import Data, SUPPORTED_MODELS, SupportedModels, \
    _replace, DEFAULT_DETECTORS, DRIFT_DETECTORS
from alibi_detect.utils.registry import registry
from alibi_detect.utils.loading import load_detector as _load_detector  # TODO: Remove in future
import warnings


# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

logger = logging.getLogger(__name__)


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
    warnings.warn("In a future version, this function will be moved to "
                  "alibi_detect.loading.load_detector()", DeprecationWarning)
    return _load_detector(filepath, **kwargs)


# TODO - eventually this will become save_detector (once outlier and adversarial updated to save via config.tonl)
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
        preprocess_cfg = _save_preprocess(preprocess_fn, backend, cfg['input_shape'], filepath, verbose)
        cfg['preprocess_fn'] = preprocess_cfg

    # Serialize kernel
    kernel = cfg.get('kernel', None)
    if kernel is not None:
        device = detector.device.type if hasattr(detector, 'device') else None
        cfg['kernel'] = _save_kernel(kernel, filepath, device, verbose)
        if isinstance(kernel, dict):  # serialise proj from DeepKernel
            cfg['kernel']['proj'], _ = _save_model(kernel['proj'], base_path=filepath, input_shape=cfg['input_shape'],
                                                  backend=backend, verbose=verbose)

    # ClassifierDrift and SpotTheDiffDrift specific artefacts.
    # Serialize detector model
    model = cfg.get('model', None)
    if model is not None:
        model_cfg, _ = _save_model(model, base_path=filepath, input_shape=cfg['input_shape'],
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
    save_config(cfg, filepath)


def save_config(cfg: dict, filepath: Union[str, os.PathLike]) -> dict:
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)
    cfg = _resolve_paths(cfg)
    cfg = _replace(cfg, None, "None")  # Note: None replaced with "None" as None/null not valid TOML
    with open(filepath.joinpath('config.toml'), 'w') as f:
        toml.dump(cfg, f, encoder=toml.TomlNumpyEncoder())
    return cfg


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
                  save_format: Literal['tf', 'h5'] = 'h5') -> None:  # TODO - change to tf, later PR
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


def _save_preprocess(preprocess_fn: Callable,
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
            cfg_model, cfg_embed = _save_model(v, filepath, input_shape, backend, local_path, verbose)
            kwargs.update({k: cfg_model})
            if cfg_embed is not None:
                kwargs.update({'embedding': cfg_embed})

        # Tokenizer
        elif isinstance(v, PreTrainedTokenizerBase):
            cfg_token = _save_tokenizer(v, filepath, local_path, verbose)
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
        src = str(local_path.with_suffix('.dill'))

    return src, kwargs


def _save_embedding(embed: tf.keras.Model,
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


def _save_model(model: SUPPORTED_MODELS,
               base_path: Path,
               input_shape: tuple,
               backend: str,
               path: Path = Path('.'),
               verbose: bool = False) -> Tuple[dict, Optional[dict]]:
    filepath = base_path.joinpath(path)

    if backend == 'tensorflow':
        cfg_model, cfg_embed = {}, None
        if isinstance(model, UAE):
            if isinstance(model.encoder.layers[0], TransformerEmbedding):  # text drift
                # embedding
                cfg_embed = {}
                embed = model.encoder.layers[0].model
                cfg_embed.update({'type': model.encoder.layers[0].emb_type})
                cfg_embed.update({'layers': model.encoder.layers[0].hs_emb.keywords['layers']})
                _save_embedding(embed, cfg_embed, filepath.joinpath('embedding'))
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

        save_tf_model(model, filepath=filepath, save_dir='model')

    else:
        raise NotImplementedError("Saving of pytorch models is not yet implemented.")

    cfg_model.update({'src': path.joinpath('model')})
    return cfg_model, cfg_embed


def _save_tokenizer(tokenizer: PreTrainedTokenizerBase,
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


def _save_kernel(kernel: Callable,
                filepath: Path,
                device: Optional[str],
                filename: str = 'kernel.dill',
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
    filename
        Filename to save to (if the kernel is a generic callable).
    verbose
        Whether to print progress messages.

    Returns
    -------
    The kernel config dictionary.
    """
    cfg_kernel = {}

    keys = [k for k, v in registry.get_all().items() if type(kernel) == v or kernel == v]
    registry_str = keys[0] if len(keys) == 1 else None
    if registry_str is not None:  # alibi-detect registered kernel
        cfg_kernel.update({'src': '@' + registry_str})

        # kwargs for registered kernel
        sigma = kernel.sigma if hasattr(kernel, 'sigma') else None
        sigma = sigma.cpu() if device == 'cuda' else sigma
        cfg_kernel.update({
            'sigma': sigma.numpy().tolist(),
            'trainable': kernel.trainable if hasattr(kernel, 'trainable') else None
        })

    elif isinstance(kernel, dict):  # DeepKernel config dict
        kernel_a = _save_kernel(kernel['kernel_a'], filepath, device, filename='kernel_a.dill', verbose=verbose)
        kernel_b = kernel.get('kernel_b')
        if kernel_b is not None:
            kernel_b = _save_kernel(kernel['kernel_b'], filepath, device, filename='kernel_b.dill', verbose=verbose)
        cfg_kernel.update({
            'kernel_a': kernel_a,
            'kernel_b': kernel_b,
            'proj': kernel['proj'],
            'eps': kernel['eps']
        })

    elif isinstance(kernel, Callable):  # generic callable
        with open(filepath.joinpath(filename), 'wb') as f:
            dill.dump(kernel, f)
        cfg_kernel.update({'src': filename})

    else:  # kernel could not be saved
        raise ValueError("Could not save kernel. Is it a valid Callable or a alibi-detect registered kernel?")

    return cfg_kernel
