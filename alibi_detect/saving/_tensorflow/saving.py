import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dill  # dispatch table setting not done here as done in top-level saving.py file
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer

# Below imports are used for legacy saving, and will be removed (or moved to utils/loading.py) in the future
from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.cd import (ChiSquareDrift, ClassifierDrift, KSDrift,
                             MMDDrift, TabularDrift)
from alibi_detect.cd.tensorflow import UAE, HiddenOutput
from alibi_detect.cd.tensorflow.classifier import ClassifierDriftTF
from alibi_detect.cd.tensorflow.mmd import MMDDriftTF
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.od import (LLR, IForest, Mahalanobis, OutlierAE,
                             OutlierAEGMM, OutlierProphet, OutlierSeq2Seq,
                             OutlierVAE, OutlierVAEGMM, SpectralResidual)
from alibi_detect.utils._types import Literal
from alibi_detect.utils.tensorflow.kernels import GaussianRBF
from alibi_detect.utils.missing_optional_dependency import MissingDependency
from alibi_detect.utils.frameworks import Framework

logger = logging.getLogger(__name__)


def save_model_config(model: Callable,
                      base_path: Path,
                      input_shape: Optional[tuple],
                      local_path: Path = Path('.')) -> Tuple[dict, Optional[dict]]:
    """
    Save a TensorFlow model to a config dictionary. When a model has a text embedding model contained within it,
    this is extracted and saved separately.

    Parameters
    ----------
    model
        The model to save.
    base_path
        Base filepath to save to (the location of the `config.toml` file).
    input_shape
        The input dimensions of the model (after the optional embedding has been applied).
    local_path
        A local (relative) filepath to append to base_path.

    Returns
    -------
    A tuple containing the model and embedding config dicts.
    """
    cfg_model: Optional[Dict[str, Any]] = None
    cfg_embed: Optional[Dict[str, Any]] = None
    if isinstance(model, UAE):
        if isinstance(model.encoder.layers[0], TransformerEmbedding):  # if UAE contains embedding and encoder
            if input_shape is None:
                raise ValueError('Cannot save combined embedding and model when `input_shape` is None.')

            # embedding
            embed = model.encoder.layers[0]
            cfg_embed = save_embedding_config(embed, base_path, local_path.joinpath('embedding'))
            # preprocessing encoder
            inputs = Input(shape=input_shape, dtype=tf.int64)
            model.encoder.call(inputs)
            shape_enc = (model.encoder.layers[0].output.shape[-1],)
            layers = [InputLayer(input_shape=shape_enc)] + model.encoder.layers[1:]
            model = tf.keras.Sequential(layers)
            _ = model(tf.zeros((1,) + shape_enc))
        else:  # If UAE is simply an encoder
            model = model.encoder
    elif isinstance(model, TransformerEmbedding):
        cfg_embed = save_embedding_config(model, base_path, local_path.joinpath('embedding'))
        model = None
    elif isinstance(model, HiddenOutput):
        model = model.model
    elif isinstance(model, tf.keras.Model):  # Last as TransformerEmbedding and UAE are tf.keras.Model's
        model = model
    else:
        raise ValueError('Model not recognised, cannot save.')

    if model is not None:
        filepath = base_path.joinpath(local_path)
        save_model(model, filepath=filepath.joinpath('model'))
        cfg_model = {
            'flavour': Framework.TENSORFLOW.value,
            'src': local_path.joinpath('model')
        }
    return cfg_model, cfg_embed


def save_model(model: tf.keras.Model,
               filepath: Union[str, os.PathLike],
               filename: str = 'model',
               save_format: Literal['tf', 'h5'] = 'h5') -> None:  # TODO - change to tf, later PR
    """
    Save TensorFlow model.

    Parameters
    ----------
    model
        The tf.keras.Model to save.
    filepath
        Save directory.
    filename
        Name of file to save to within the filepath directory.
    save_format
        The format to save to. 'tf' to save to the newer SavedModel format, 'h5' to save to the lighter-weight
        legacy hdf5 format.
    """
    # create folder to save model in
    model_path = Path(filepath)
    if not model_path.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(model_path))
        model_path.mkdir(parents=True, exist_ok=True)

    # save model
    model_path = model_path.joinpath(filename + '.h5') if save_format == 'h5' else model_path

    if isinstance(model, tf.keras.Model):
        model.save(model_path, save_format=save_format)
    else:
        raise ValueError('The extracted model to save is not a `tf.keras.Model`. Cannot save.')


def save_embedding_config(embed: TransformerEmbedding,
                          base_path: Path,
                          local_path: Path = Path('.')) -> dict:
    """
    Save embeddings for text drift models.

    Parameters
    ----------
    embed
        Embedding model.
    base_path
        Base filepath to save to (the location of the `config.toml` file).
    local_path
        A local (relative) filepath to append to base_path.
    """
    # create folder to save model in
    filepath = base_path.joinpath(local_path)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # Populate config dict
    cfg_embed: Dict[str, Any] = {}
    cfg_embed.update({'type': embed.emb_type})
    cfg_embed.update({'layers': embed.hs_emb.keywords['layers']})
    cfg_embed.update({'src': local_path})
    cfg_embed.update({'flavour': Framework.TENSORFLOW.value})

    # Save embedding model
    logger.info('Saving embedding model to {}.'.format(filepath))
    embed.model.save_pretrained(filepath)

    return cfg_embed


def save_optimizer_config(optimizer: Union['tf.keras.optimizers.Optimizer', 'tf.keras.optimizers.legacy.Optimizer']):
    """

    Parameters
    ----------
    optimizer
        The tensorflow optimizer to serialize.

    Returns
    -------
    The tensorflow optimizer's config dictionary.
    """
    return tf.keras.optimizers.serialize(optimizer)


#######################################################################################################
# TODO: Everything below here is legacy saving code, and will be removed in the future
#######################################################################################################
def save_embedding_legacy(embed: TransformerEmbedding,
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


def save_detector_legacy(detector, filepath):
    detector_name = detector.meta['name']

    # save metadata
    logger.info('Saving metadata and detector to {}'.format(filepath))

    with open(filepath.joinpath('meta.dill'), 'wb') as f:
        dill.dump(detector.meta, f)

    # save detector specific parameters
    if isinstance(detector, OutlierAE):
        state_dict = state_ae(detector)
    elif isinstance(detector, OutlierVAE):
        state_dict = state_vae(detector)
    elif isinstance(detector, Mahalanobis):
        state_dict = state_mahalanobis(detector)
    elif isinstance(detector, IForest):
        state_dict = state_iforest(detector)
    elif isinstance(detector, ChiSquareDrift):
        state_dict, model, embed, embed_args, tokenizer = state_chisquaredrift(detector)
    elif isinstance(detector, ClassifierDrift):
        state_dict, clf_drift, model, embed, embed_args, tokenizer = state_classifierdrift(detector)
    elif isinstance(detector, TabularDrift):
        state_dict, model, embed, embed_args, tokenizer = state_tabulardrift(detector)
    elif isinstance(detector, KSDrift):
        state_dict, model, embed, embed_args, tokenizer = state_ksdrift(detector)
    elif isinstance(detector, MMDDrift):
        state_dict, model, embed, embed_args, tokenizer = state_mmddrift(detector)
    elif isinstance(detector, OutlierAEGMM):
        state_dict = state_aegmm(detector)
    elif isinstance(detector, OutlierVAEGMM):
        state_dict = state_vaegmm(detector)
    elif isinstance(detector, AdversarialAE):
        state_dict = state_adv_ae(detector)
    elif isinstance(detector, ModelDistillation):
        state_dict = state_adv_md(detector)
    elif not isinstance(OutlierProphet, MissingDependency) and isinstance(detector, OutlierProphet):
        state_dict = state_prophet(detector)
    elif isinstance(detector, SpectralResidual):
        state_dict = state_sr(detector)
    elif isinstance(detector, OutlierSeq2Seq):
        state_dict = state_s2s(detector)
    elif isinstance(detector, LLR):
        state_dict = state_llr(detector)
    else:
        raise NotImplementedError('The %s detector does not have a legacy save method.' % detector_name)

    with open(filepath.joinpath(detector_name + '.dill'), 'wb') as f:
        dill.dump(state_dict, f)

    # save detector specific TensorFlow models
    model_dir = filepath.joinpath('model')
    if isinstance(detector, OutlierAE):
        save_tf_ae(detector, filepath)
    elif isinstance(detector, OutlierVAE):
        save_tf_vae(detector, filepath)
    elif isinstance(detector, (ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, TabularDrift)):
        if model is not None:
            save_model(model, model_dir, filename='encoder')
        if embed is not None:
            save_embedding_legacy(embed, embed_args, filepath)
        if tokenizer is not None:
            tokenizer.save_pretrained(filepath.joinpath('model'))
        if detector_name == 'ClassifierDriftTF':
            save_model(clf_drift, model_dir, filename='clf_drift')
    elif isinstance(detector, OutlierAEGMM):
        save_tf_aegmm(detector, filepath)
    elif isinstance(detector, OutlierVAEGMM):
        save_tf_vaegmm(detector, filepath)
    elif isinstance(detector, AdversarialAE):
        save_tf_ae(detector, filepath)
        save_model(detector.model, model_dir)
        save_tf_hl(detector.model_hl, filepath)
    elif isinstance(detector, ModelDistillation):
        save_model(detector.distilled_model, model_dir, filename='distilled_model')
        save_model(detector.model, model_dir, filename='model')
    elif isinstance(detector, OutlierSeq2Seq):
        save_tf_s2s(detector, filepath)
    elif isinstance(detector, LLR):
        save_tf_llr(detector, filepath)


def preprocess_step_drift(cd: Union[ChiSquareDrift, ClassifierDriftTF, KSDrift, MMDDriftTF, TabularDrift]) \
        -> Tuple[
            Optional[Callable], Dict, Optional[tf.keras.Model],
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
            elif isinstance(v, tf.keras.Model):
                model = v
                preprocess_kwargs['model'] = 'custom'
            elif hasattr(v, '__module__'):
                if 'transformers' in v.__module__:  # transformers tokenizer
                    tokenizer = v
                    preprocess_kwargs[k] = v.__module__
            else:
                preprocess_kwargs[k] = v
    elif callable(cd.preprocess_fn):
        preprocess_fn = cd.preprocess_fn
    return preprocess_fn, preprocess_kwargs, model, embed, embed_args, tokenizer, load_emb


def state_chisquaredrift(cd: ChiSquareDrift) -> Tuple[
            Dict, Optional[tf.keras.Model],
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
                'x_ref_preprocessed': True,
                'preprocess_at_init': cd.preprocess_at_init,
                'update_x_ref': cd.update_x_ref,
                'correction': cd.correction,
                'n_features': cd.n_features,
                'input_shape': cd.input_shape,
            },
        'other':
            {
                'n': cd.n,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
    }
    return state_dict, model, embed, embed_args, tokenizer


def state_classifierdrift(cd: ClassifierDrift) -> Tuple[
            Dict, tf.keras.Model,
            Optional[tf.keras.Model],
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
                'x_ref_preprocessed': True,
                'preprocess_at_init': cd._detector.preprocess_at_init,
                'update_x_ref': cd._detector.update_x_ref,
                'preds_type': cd._detector.preds_type,
                'binarize_preds': cd._detector.binarize_preds,
                'train_size': cd._detector.train_size,
                'train_kwargs': cd._detector.train_kwargs,
            },
        'other':
            {
                'n': cd._detector.n,
                'skf': cd._detector.skf,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
    }
    return state_dict, cd._detector.model, model, embed, embed_args, tokenizer


def state_tabulardrift(cd: TabularDrift) -> Tuple[
            Dict, Optional[tf.keras.Model],
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
                'x_ref_preprocessed': True,
                'preprocess_at_init': cd.preprocess_at_init,
                'update_x_ref': cd.update_x_ref,
                'correction': cd.correction,
                'alternative': cd.alternative,
                'n_features': cd.n_features,
                'input_shape': cd.input_shape,
            },
        'other':
            {
                'n': cd.n,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
    }
    return state_dict, model, embed, embed_args, tokenizer


def state_ksdrift(cd: KSDrift) -> Tuple[
            Dict, Optional[tf.keras.Model],
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
                'x_ref_preprocessed': True,
                'preprocess_at_init': cd.preprocess_at_init,
                'update_x_ref': cd.update_x_ref,
                'correction': cd.correction,
                'alternative': cd.alternative,
                'n_features': cd.n_features,
                'input_shape': cd.input_shape,
            },
        'other':
            {
                'n': cd.n,
                'load_text_embedding': load_emb,
                'preprocess_fn': preprocess_fn,
                'preprocess_kwargs': preprocess_kwargs
            }
    }
    return state_dict, model, embed, embed_args, tokenizer


def state_mmddrift(cd: MMDDrift) -> Tuple[
            Dict, Optional[tf.keras.Model],
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
                'x_ref_preprocessed': True,
                'preprocess_at_init': cd._detector.preprocess_at_init,
                'update_x_ref': cd._detector.update_x_ref,
                'sigma': sigma,
                'configure_kernel_from_x_ref': not cd._detector.infer_sigma,
                'n_permutations': cd._detector.n_permutations,
                'input_shape': cd._detector.input_shape,
            },
        'other':
            {
                'n': cd._detector.n,
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
