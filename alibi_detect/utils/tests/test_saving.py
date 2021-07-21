from functools import partial
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
import sys
from tempfile import TemporaryDirectory
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from typing import Callable
from alibi_detect.ad import AdversarialAE, ModelDistillation
from alibi_detect.cd import ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, TabularDrift
from alibi_detect.cd.tensorflow import UAE, preprocess_drift
from alibi_detect.models.tensorflow.autoencoder import DecoderLSTM, EncoderLSTM
from alibi_detect.od import (IForest, LLR, Mahalanobis, OutlierAEGMM, OutlierVAE, OutlierVAEGMM,
                             OutlierProphet, SpectralResidual, OutlierSeq2Seq, OutlierAE)
from alibi_detect.utils.saving import save_detector, load_detector  # type: ignore

input_dim = 4
latent_dim = 2
n_gmm = 2
threshold = 10.
threshold_drift = .55
n_folds_drift = 5
samples = 6
seq_len = 10
p_val = .05
X_ref = np.random.rand(samples * input_dim).reshape(samples, input_dim)
X_ref_cat = np.tile(np.array([np.arange(samples)] * input_dim).T, (2, 1))
X_ref_mix = X_ref.copy()
X_ref_mix[:, 0] = np.tile(np.array(np.arange(samples // 2)), (1, 2)).T[:, 0]
n_permutations = 10

# define encoder and decoder
encoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(input_dim,)),
        Dense(5, activation=tf.nn.relu),
        Dense(latent_dim, activation=None)
    ]
)

decoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(latent_dim,)),
        Dense(5, activation=tf.nn.relu),
        Dense(input_dim, activation=tf.nn.sigmoid)
    ]
)

kwargs = {'encoder_net': encoder_net,
          'decoder_net': decoder_net}

preprocess_fn = partial(preprocess_drift, model=UAE(encoder_net=encoder_net))

gmm_density_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(latent_dim + 2,)),
        Dense(10, activation=tf.nn.relu),
        Dense(n_gmm, activation=tf.nn.softmax)
    ]
)

threshold_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(seq_len, latent_dim)),
        Dense(5, activation=tf.nn.relu)
    ]
)

# define model
inputs = tf.keras.Input(shape=(input_dim,))
outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

detector = [
    AdversarialAE(threshold=threshold,
                  model=model,
                  **kwargs),
    ModelDistillation(threshold=threshold,
                      model=model,
                      distilled_model=model),
    IForest(threshold=threshold),
    LLR(threshold=threshold, model=model),
    Mahalanobis(threshold=threshold),
    OutlierAEGMM(threshold=threshold,
                 gmm_density_net=gmm_density_net,
                 n_gmm=n_gmm,
                 **kwargs),
    OutlierVAE(threshold=threshold,
               latent_dim=latent_dim,
               samples=samples,
               **kwargs),
    OutlierAE(threshold=threshold,
              **kwargs),
    OutlierVAEGMM(threshold=threshold,
                  gmm_density_net=gmm_density_net,
                  n_gmm=n_gmm,
                  latent_dim=latent_dim,
                  samples=samples,
                  **kwargs),
    OutlierProphet(threshold=.7,
                   growth='logistic'),
    SpectralResidual(threshold=threshold,
                     window_amp=10,
                     window_local=10),
    OutlierSeq2Seq(input_dim,
                   seq_len,
                   threshold=threshold,
                   threshold_net=threshold_net,
                   latent_dim=latent_dim),
    KSDrift(X_ref,
            p_val=p_val,
            preprocess_x_ref=False,
            preprocess_fn=preprocess_fn),
    MMDDrift(X_ref,
             p_val=p_val,
             preprocess_x_ref=False,
             preprocess_fn=preprocess_fn,
             configure_kernel_from_x_ref=True,
             n_permutations=n_permutations),
    ChiSquareDrift(X_ref_cat,
                   p_val=p_val,
                   preprocess_x_ref=True),
    TabularDrift(X_ref_mix,
                 p_val=p_val,
                 categories_per_feature={0: None},
                 preprocess_x_ref=True),
    ClassifierDrift(X_ref,
                    model=model,
                    p_val=p_val,
                    n_folds=n_folds_drift,
                    train_size=None)
]
n_tests = len(detector)


@pytest.fixture
def select_detector(request):
    return detector[request.param]


@pytest.mark.parametrize('select_detector', list(range(n_tests)), indirect=True)
def test_save_load(select_detector):
    det = select_detector
    det_name = det.meta['name']

    # save and load functionality does not work for OutlierProphet and Python 3.6.
    # https://github.com/facebook/prophet/issues/1361
    if sys.version_info.minor == 6 and isinstance(det, OutlierProphet):
        return

    with TemporaryDirectory() as temp_dir:
        temp_dir += '/'
        save_detector(det, temp_dir)
        det_load = load_detector(temp_dir)
        det_load_name = det_load.meta['name']
        assert det_load_name == det_name

        if not type(det_load) in [
            OutlierProphet, ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, TabularDrift
        ]:
            assert det_load.threshold == det.threshold == threshold

        if type(det_load) in [OutlierVAE, OutlierVAEGMM]:
            assert det_load.samples == det.samples == samples

        if type(det_load) == AdversarialAE or type(det_load) == ModelDistillation:
            for layer in det_load.model.layers:
                assert not layer.trainable

        if type(det_load) == OutlierAEGMM:
            assert isinstance(det_load.aegmm.encoder, tf.keras.Sequential)
            assert isinstance(det_load.aegmm.decoder, tf.keras.Sequential)
            assert isinstance(det_load.aegmm.gmm_density, tf.keras.Sequential)
            assert isinstance(det_load.aegmm, tf.keras.Model)
            assert det_load.aegmm.n_gmm == n_gmm
        elif type(det_load) == OutlierVAEGMM:
            assert isinstance(det_load.vaegmm.encoder.encoder_net, tf.keras.Sequential)
            assert isinstance(det_load.vaegmm.decoder, tf.keras.Sequential)
            assert isinstance(det_load.vaegmm.gmm_density, tf.keras.Sequential)
            assert isinstance(det_load.vaegmm, tf.keras.Model)
            assert det_load.vaegmm.latent_dim == latent_dim
            assert det_load.vaegmm.n_gmm == n_gmm
        elif type(det_load) in [AdversarialAE, OutlierAE]:
            assert isinstance(det_load.ae.encoder.encoder_net, tf.keras.Sequential)
            assert isinstance(det_load.ae.decoder.decoder_net, tf.keras.Sequential)
            assert isinstance(det_load.ae, tf.keras.Model)
        elif type(det_load) == ModelDistillation:
            assert isinstance(det_load.model, tf.keras.Sequential) or isinstance(det_load.model, tf.keras.Model)
            assert (isinstance(det_load.distilled_model, tf.keras.Sequential) or
                    isinstance(det_load.distilled_model, tf.keras.Model))
        elif type(det_load) == OutlierVAE:
            assert isinstance(det_load.vae.encoder.encoder_net, tf.keras.Sequential)
            assert isinstance(det_load.vae.decoder.decoder_net, tf.keras.Sequential)
            assert isinstance(det_load.vae, tf.keras.Model)
            assert det_load.vae.latent_dim == latent_dim
        elif type(det_load) == Mahalanobis:
            assert det_load.clip is None
            assert det_load.mean == det_load.C == det_load.n == 0
            assert det_load.meta['detector_type'] == 'online'
        elif type(det_load) == OutlierProphet:
            assert det_load.model.interval_width == .7
            assert det_load.model.growth == 'logistic'
            assert det_load.meta['data_type'] == 'time-series'
        elif type(det_load) == SpectralResidual:
            assert det_load.window_amp == 10
            assert det_load.window_local == 10
        elif type(det_load) == OutlierSeq2Seq:
            assert isinstance(det_load.seq2seq, tf.keras.Model)
            assert isinstance(det_load.seq2seq.threshold_net, tf.keras.Sequential)
            assert isinstance(det_load.seq2seq.encoder, EncoderLSTM)
            assert isinstance(det_load.seq2seq.decoder, DecoderLSTM)
            assert det_load.latent_dim == latent_dim
            assert det_load.threshold == threshold
            assert det_load.shape == (-1, seq_len, input_dim)
        elif type(det_load) == KSDrift:
            assert det_load.n_features == latent_dim
            assert det_load.p_val == p_val
            assert (det_load.x_ref == X_ref).all()
            assert isinstance(det_load.preprocess_fn, Callable)
            assert det_load.preprocess_fn.func.__name__ == 'preprocess_drift'
        elif type(det_load) in [ChiSquareDrift, TabularDrift]:
            assert isinstance(det_load.x_ref_categories, dict)
            assert det_load.p_val == p_val
            x = X_ref_cat.copy() if isinstance(det_load, ChiSquareDrift) else X_ref_mix.copy()
            assert (det_load.x_ref == x).all()
        elif type(det_load) == MMDDrift:
            assert not det_load._detector.infer_sigma
            assert det_load._detector.n_permutations == n_permutations
            assert det_load._detector.p_val == p_val
            assert (det_load._detector.x_ref == X_ref).all()
            assert isinstance(det_load._detector.preprocess_fn, Callable)
            assert det_load._detector.preprocess_fn.func.__name__ == 'preprocess_drift'
        elif type(det_load) == ClassifierDrift:
            assert det_load._detector.p_val == p_val
            assert (det_load._detector.x_ref == X_ref).all()
            assert isinstance(det_load._detector.skf, StratifiedKFold)
            assert isinstance(det_load._detector.train_kwargs, dict)
            assert isinstance(det_load._detector.model, tf.keras.Model)
        elif type(det_load) == LLR:
            assert isinstance(det_load.dist_s, tf.keras.Model)
            assert isinstance(det_load.dist_b, tf.keras.Model)
            assert not det_load.sequential
            assert not det_load.has_log_prob
