from pathlib import Path
from typing import Callable, Tuple, Optional, Dict, Union, List, Any
from alibi_detect.utils._types import Literal
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer
from alibi_detect.cd.tensorflow import HiddenOutput, UAE
from alibi_detect.models.tensorflow import TransformerEmbedding
import os
import logging
import dill  # dispatch table setting not done here as done in top-level saving.py file

# Below imports are used for legacy saving, and will be removed (or moved to utils/loading.py) in the future
from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.od import (IForest, LLR, Mahalanobis, OutlierAE, OutlierAEGMM, OutlierProphet,
                             OutlierSeq2Seq, OutlierVAE, OutlierVAEGMM, SpectralResidual)

logger = logging.getLogger(__name__)


def save_model_config(model: Callable,
                      base_path: Path,
                      input_shape: tuple,
                      path: Path = Path('.')) -> Tuple[dict, Optional[dict]]:
    """
    Save a model to a config dictionary. When a model has a text embedding model contained within it,
    this is extracted and saved separately.

    Parameters
    ----------
    model
        The model to save.
    base_path
        Base filepath to save to.
    input_shape
        The input dimensions of the model (after the optional embedding has been applied).
    path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    A tuple containing the model and embedding config dicts.
    """

    filepath = base_path.joinpath(path)

    cfg_model = {}  # type: Dict[str, Any]
    cfg_embed = None  # type: Optional[Dict[str, Any]]
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
    cfg_model.update({'src': path.joinpath('model')})
    return cfg_model, cfg_embed


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


# Note: save_embedding is backend agnostic, but can't be put in utils/saving.py due to circular import.
#  Hence it will need to be duplicated in utils/pytorch/_saving.py, or moved to a 3rd file.
def _save_embedding(embed: TransformerEmbedding,
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
    logger.info('Saving embedding model to {}.'.format(filepath.joinpath('embedding.dill')))
    embed.save_pretrained(filepath)
    with open(filepath.joinpath('embedding.dill'), 'wb') as f:
        dill.dump(embed_args, f)


#######################################################################################################
# TODO: Everything below here is legacy saving code, and will be removed in the future
#######################################################################################################
def save_detector_legacy(detector, filepath):
    detector_name = detector.__class__.__name__

    # save metadata
    logger.info('Saving metadata and detector to {}'.format(filepath))

    with open(filepath.joinpath('meta.dill'), 'wb') as f:
        dill.dump(detector.meta, f)

    # save outlier detector specific parameters
    if isinstance(detector, OutlierAE):
        state_dict = _state_ae(detector)
    elif isinstance(detector, OutlierVAE):
        state_dict = _state_vae(detector)
    elif isinstance(detector, Mahalanobis):
        state_dict = _state_mahalanobis(detector)
    elif isinstance(detector, IForest):
        state_dict = _state_iforest(detector)
    elif isinstance(detector, OutlierAEGMM):
        state_dict = _state_aegmm(detector)
    elif isinstance(detector, OutlierVAEGMM):
        state_dict = _state_vaegmm(detector)
    elif isinstance(detector, AdversarialAE):
        state_dict = _state_adv_ae(detector)
    elif isinstance(detector, ModelDistillation):
        state_dict = _state_adv_md(detector)
    elif isinstance(detector, OutlierProphet):
        state_dict = _state_prophet(detector)
    elif isinstance(detector, SpectralResidual):
        state_dict = _state_sr(detector)
    elif isinstance(detector, OutlierSeq2Seq):
        state_dict = _state_s2s(detector)
    elif isinstance(detector, LLR):
        state_dict = _state_llr(detector)

    with open(filepath.joinpath(detector_name + '.dill'), 'wb') as f:
        dill.dump(state_dict, f)

    # save outlier detector specific TensorFlow models
    if isinstance(detector, OutlierAE):
        save_tf_ae(detector, filepath)
    elif isinstance(detector, OutlierVAE):
        save_tf_vae(detector, filepath)
    elif isinstance(detector, OutlierAEGMM):
        save_tf_aegmm(detector, filepath)
    elif isinstance(detector, OutlierVAEGMM):
        save_tf_vaegmm(detector, filepath)
    elif isinstance(detector, AdversarialAE):
        save_tf_ae(detector, filepath)
        save_tf_model(detector.model, filepath)
        save_tf_hl(detector.model_hl, filepath)
    elif isinstance(detector, ModelDistillation):
        save_tf_model(detector.distilled_model, filepath, save_dir='distilled_model')
        save_tf_model(detector.model, filepath, save_dir='model')
    elif isinstance(detector, OutlierSeq2Seq):
        save_tf_s2s(detector, filepath)
    elif isinstance(detector, LLR):
        save_tf_llr(detector, filepath)


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
