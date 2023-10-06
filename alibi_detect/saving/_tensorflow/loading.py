import logging
import os
from importlib import import_module
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Type

import dill
import tensorflow as tf
from tensorflow_probability.python.distributions.distribution import \
    Distribution
from transformers import AutoTokenizer

from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.ad.adversarialae import DenseHidden
from alibi_detect.cd import (ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, TabularDrift)
from alibi_detect.cd.tensorflow import UAE, HiddenOutput
from alibi_detect.cd.tensorflow.preprocess import _Encoder
from alibi_detect.models.tensorflow import PixelCNN, TransformerEmbedding
from alibi_detect.models.tensorflow.autoencoder import (AE, AEGMM, VAE, VAEGMM,
                                                        DecoderLSTM,
                                                        EncoderLSTM, Seq2Seq)
from alibi_detect.od import (LLR, IForest, Mahalanobis, OutlierAE,
                             OutlierAEGMM, OutlierProphet, OutlierSeq2Seq,
                             OutlierVAE, OutlierVAEGMM, SpectralResidual)
from alibi_detect.od.llr import build_model
from alibi_detect.utils.tensorflow.kernels import DeepKernel
from alibi_detect.utils.frameworks import Framework
# Below imports are used for legacy loading, and will be removed (or moved to utils/loading.py) in the future
from alibi_detect.version import __version__
from alibi_detect.base import Detector
from alibi_detect.saving._typing import VALID_DETECTORS

logger = logging.getLogger(__name__)


def load_model(filepath: Union[str, os.PathLike],
               filename: str = 'model',
               custom_objects: dict = None,
               layer: Optional[int] = None,
               ) -> tf.keras.Model:
    """
    Load TensorFlow model.

    Parameters
    ----------
    filepath
        Saved model directory.
    filename
        Name of saved model within the filepath directory.
    custom_objects
        Optional custom objects when loading the TensorFlow model.
    layer
        Optional index of a hidden layer to extract. If not `None`, a
        :py:class:`~alibi_detect.cd.tensorflow.HiddenOutput` model is returned.

    Returns
    -------
    Loaded model.
    """
    # TODO - update this to accept tf format - later PR.
    model_dir = Path(filepath)
    model_name = filename + '.h5'
    # Check if model exists
    if model_name not in [f.name for f in model_dir.glob('[!.]*.h5')]:
        raise FileNotFoundError(f'{model_name} not found in {model_dir.resolve()}.')
    model = tf.keras.models.load_model(model_dir.joinpath(model_name), custom_objects=custom_objects)
    # Optionally extract hidden layer
    if isinstance(layer, int):
        model = HiddenOutput(model, layer=layer)
    return model


def prep_model_and_emb(model: Callable, emb: Optional[TransformerEmbedding]) -> Callable:
    """
    Function to perform final preprocessing of model (and/or embedding) before it is passed to preprocess_drift.

    Parameters
    ----------
    model
        A compatible model.
    emb
        An optional text embedding model.

    Returns
    -------
    The final model ready to passed to preprocess_drift.
    """
    # Process model (and embedding)
    model = model.encoder if isinstance(model, UAE) else model  # This is to avoid nesting UAE's already a UAE
    if emb is not None:
        model = _Encoder(emb, mlp=model)
        model = UAE(encoder_net=model)
    return model


def load_kernel_config(cfg: dict) -> Callable:
    """
    Loads a kernel from a kernel config dict.

    Parameters
    ----------
    cfg
        A kernel config dict. (see pydantic schema's).

    Returns
    -------
    The kernel.
    """
    if 'src' in cfg:  # Standard kernel config
        kernel = cfg.pop('src')
        if hasattr(kernel, 'from_config'):
            kernel = kernel.from_config(cfg)

    elif 'proj' in cfg:  # DeepKernel config
        # Kernel a
        kernel_a = cfg['kernel_a']
        kernel_b = cfg['kernel_b']
        if kernel_a != 'rbf':
            cfg['kernel_a'] = load_kernel_config(kernel_a)
        if kernel_b != 'rbf':
            cfg['kernel_b'] = load_kernel_config(kernel_b)
        # Assemble deep kernel
        kernel = DeepKernel.from_config(cfg)

    else:
        raise ValueError('Unable to process kernel. The kernel config dict must either be a `KernelConfig` with a '
                         '`src` field, or a `DeepkernelConfig` with a `proj` field.)')
    return kernel


def load_optimizer(cfg: dict) -> Union[Type['tf.keras.optimizers.Optimizer'], 'tf.keras.optimizers.Optimizer']:
    """
    Loads a TensorFlow optimzier from a optimizer config dict.

    Parameters
    ----------
    cfg
        The optimizer config dict.

    Returns
    -------
    The loaded optimizer, either as an instantiated object (if `cfg` is a tensorflow optimizer config dict), otherwise \
    as an uninstantiated class.
    """
    class_name = cfg.get('class_name')
    tf_config = cfg.get('config')
    if tf_config is not None:  # cfg is a tensorflow config dict
        return tf.keras.optimizers.deserialize(cfg)
    else:
        try:
            return getattr(import_module('tensorflow.keras.optimizers'), class_name)
        except AttributeError:
            raise ValueError(f"{class_name} is not a recognised optimizer in `tensorflow.keras.optimizers`.")


def load_embedding(src: str, embedding_type, layers) -> TransformerEmbedding:
    """
    Load a pre-trained tensorflow text embedding from a directory.
    See the `:py:class:~alibi_detect.models.tensorflow.TransformerEmbedding` documentation for a
    full description of the `embedding_type` and `layers` kwargs.

    Parameters
    ----------
    src
        Name of or path to the model.
    embedding_type
       Type of embedding to extract. Needs to be one of pooler_output,
       last_hidden_state, hidden_state or hidden_state_cls.
    layers
        A list with int's referring to the hidden layers used to extract the embedding.

    Returns
    -------
    The loaded embedding.
    """
    emb = TransformerEmbedding(src, embedding_type=embedding_type, layers=layers)
    return emb


#######################################################################################################
# TODO: Everything below here is legacy loading code, and will be removed in the future
#######################################################################################################
def load_detector_legacy(filepath: Union[str, os.PathLike], suffix: str, **kwargs) -> Detector:
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
    warnings.warn('Loading of meta.dill and meta.pickle files will be removed in a future version.', DeprecationWarning)

    if kwargs:
        k = list(kwargs.keys())
    else:
        k = []

    # check if path exists
    filepath = Path(filepath)
    if not filepath.is_dir():
        raise FileNotFoundError(f'{filepath} does not exist.')

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

    if 'backend' in list(meta_dict.keys()) and meta_dict['backend'] == Framework.PYTORCH:
        raise NotImplementedError('Detectors with PyTorch backend are not yet supported.')

    detector_name = meta_dict['name']
    if detector_name not in [detector for detector in VALID_DETECTORS]:
        raise NotImplementedError(f'{detector_name} is not supported by `load_detector`.')

    # load outlier detector specific parameters
    state_dict = dill.load(open(filepath.joinpath(detector_name + suffix), 'rb'))

    # Update the drift detector preprocess kwargs if state_dict is from an old alibi-detect version (<v0.10).
    # See https://github.com/SeldonIO/alibi-detect/pull/732
    if 'kwargs' in state_dict and 'other' in state_dict:  # A drift detector if both of these exist
        if 'x_ref_preprocessed' not in state_dict['kwargs']:  # if already exists then must have been saved w/ >=v0.10
            # Set x_ref_preprocessed to True
            state_dict['kwargs']['x_ref_preprocessed'] = True
            # Move `preprocess_x_ref` from `other` to `kwargs`
            state_dict['kwargs']['preprocess_x_ref'] = state_dict['other']['preprocess_x_ref']

    # initialize detector
    model_dir = filepath.joinpath('model')
    detector: Optional[Detector] = None  # to avoid mypy errors
    if detector_name == 'OutlierAE':
        ae = load_tf_ae(filepath)
        detector = init_od_ae(state_dict, ae)
    elif detector_name == 'OutlierVAE':
        vae = load_tf_vae(filepath, state_dict)
        detector = init_od_vae(state_dict, vae)
    elif detector_name == 'Mahalanobis':
        detector = init_od_mahalanobis(state_dict)  # type: ignore[assignment]
    elif detector_name == 'IForest':
        detector = init_od_iforest(state_dict)  # type: ignore[assignment]
    elif detector_name == 'OutlierAEGMM':
        aegmm = load_tf_aegmm(filepath, state_dict)
        detector = init_od_aegmm(state_dict, aegmm)
    elif detector_name == 'OutlierVAEGMM':
        vaegmm = load_tf_vaegmm(filepath, state_dict)
        detector = init_od_vaegmm(state_dict, vaegmm)
    elif detector_name == 'AdversarialAE':
        ae = load_tf_ae(filepath)
        custom_objects = kwargs['custom_objects'] if 'custom_objects' in k else None
        model = load_model(model_dir, custom_objects=custom_objects)
        model_hl = load_tf_hl(filepath, model, state_dict)
        detector = init_ad_ae(state_dict, ae, model, model_hl)
    elif detector_name == 'ModelDistillation':
        md = load_model(model_dir, filename='distilled_model')
        custom_objects = kwargs['custom_objects'] if 'custom_objects' in k else None
        model = load_model(model_dir, custom_objects=custom_objects)
        detector = init_ad_md(state_dict, md, model)
    elif detector_name == 'OutlierProphet':
        detector = init_od_prophet(state_dict)
    elif detector_name == 'SpectralResidual':
        detector = init_od_sr(state_dict)  # type: ignore[assignment]
    elif detector_name == 'OutlierSeq2Seq':
        seq2seq = load_tf_s2s(filepath, state_dict)
        detector = init_od_s2s(state_dict, seq2seq)
    elif detector_name in ['ChiSquareDrift', 'ClassifierDriftTF', 'KSDrift', 'MMDDriftTF', 'TabularDrift']:
        emb, tokenizer = None, None
        if state_dict['other']['load_text_embedding']:
            emb, tokenizer = load_text_embed(filepath)
        try:  # legacy load_model behaviour was to return None if not found. Now it raises error, hence need try-except.
            model = load_model(model_dir, filename='encoder')
        except FileNotFoundError:
            logger.warning('No model found in {}, setting `model` to `None`.'.format(model_dir))
            model = None
        if detector_name == 'KSDrift':
            load_fn = init_cd_ksdrift
        elif detector_name == 'MMDDriftTF':
            load_fn = init_cd_mmddrift  # type: ignore[assignment]
        elif detector_name == 'ChiSquareDrift':
            load_fn = init_cd_chisquaredrift  # type: ignore[assignment]
        elif detector_name == 'TabularDrift':
            load_fn = init_cd_tabulardrift  # type: ignore[assignment]
        elif detector_name == 'ClassifierDriftTF':
            # Don't need try-except here since model is not optional for ClassifierDrift
            clf_drift = load_model(model_dir, filename='clf_drift')
            load_fn = partial(init_cd_classifierdrift, clf_drift)  # type: ignore[assignment]
        else:
            raise NotImplementedError
        detector = load_fn(state_dict, model, emb, tokenizer, **kwargs)  # type: ignore[assignment]
    elif detector_name == 'LLR':
        models = load_tf_llr(filepath, **kwargs)
        detector = init_od_llr(state_dict, models)
    else:
        raise NotImplementedError

    # TODO - add tests back in!

    detector.meta = meta_dict
    logger.info('Finished loading detector.')
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
    """Legacy function to load text embedding."""
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
    """Return preprocessing function and kwargs."""
    if kwargs:  # override defaults
        keys = list(kwargs.keys())
        preprocess_fn = kwargs['preprocess_fn'] if 'preprocess_fn' in keys else None
        preprocess_kwargs = kwargs['preprocess_kwargs'] if 'preprocess_kwargs' in keys else None
        return preprocess_fn, preprocess_kwargs
    elif model is not None and callable(state_dict['preprocess_fn']) \
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
    if callable(preprocess_fn) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    state_dict['kwargs']['train_kwargs']['optimizer'] = \
        tf.keras.optimizers.get(state_dict['kwargs']['train_kwargs']['optimizer'])
    args = list(state_dict['args'].values()) + [clf_drift]
    cd = ClassifierDrift(*args, **state_dict['kwargs'])
    attrs = state_dict['other']
    cd._detector.n = attrs['n']
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
    if callable(preprocess_fn) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = ChiSquareDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    attrs = state_dict['other']
    cd.n = attrs['n']
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
    if callable(preprocess_fn) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = TabularDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    attrs = state_dict['other']
    cd.n = attrs['n']
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
    if callable(preprocess_fn) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = KSDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    attrs = state_dict['other']
    cd.n = attrs['n']
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
    if callable(preprocess_fn) and isinstance(preprocess_kwargs, dict):
        state_dict['kwargs'].update({'preprocess_fn': partial(preprocess_fn, **preprocess_kwargs)})
    cd = MMDDrift(*list(state_dict['args'].values()), **state_dict['kwargs'])
    attrs = state_dict['other']
    cd._detector.n = attrs['n']
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
    models
        Tuple containing the model and background model.

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
