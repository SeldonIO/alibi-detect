import numpy as np
import tensorflow as tf
from alibi_detect.models.losses import elbo, loss_adv_vae, loss_aegmm, loss_vaegmm

N, K, D, F = 10, 5, 1, 3
x = np.random.rand(N, F).astype(np.float32)
y = np.random.rand(N, F).astype(np.float32)
sim = 1.
cov_diag = tf.ones(x.shape[1])
cov_full = tf.eye(x.shape[1])


def test_elbo():
    assert elbo(x, y, cov_full=cov_full) == elbo(x, y, cov_diag=cov_diag) == elbo(x, y, sim=sim)
    assert elbo(x, y, sim=.05).numpy() > 0
    assert elbo(x, x, sim=.05).numpy() < 0


z = np.random.rand(N, D).astype(np.float32)
gamma = np.random.rand(N, K).astype(np.float32)


def test_loss_aegmm():
    loss = loss_aegmm(x, y, z, gamma, w_energy=.1, w_cov_diag=.005)
    loss_no_cov = loss_aegmm(x, y, z, gamma, w_energy=.1, w_cov_diag=0.)
    loss_xx = loss_aegmm(x, x, z, gamma, w_energy=.1, w_cov_diag=0.)
    assert loss > loss_no_cov
    assert loss_no_cov > loss_xx


def test_loss_vaegmm():
    loss = loss_vaegmm(x, y, z, gamma, w_recon=1e-7, w_energy=.1, w_cov_diag=.005)
    loss_no_recon = loss_vaegmm(x, y, z, gamma, w_recon=0., w_energy=.1, w_cov_diag=.005)
    loss_no_recon_cov = loss_vaegmm(x, y, z, gamma, w_recon=0., w_energy=.1, w_cov_diag=0.)
    loss_xx = loss_vaegmm(x, x, z, gamma, w_recon=1e-7, w_energy=.1, w_cov_diag=.005)
    assert loss > loss_no_recon
    assert loss_no_recon > loss_no_recon_cov
    assert loss > loss_xx


inputs = tf.keras.Input(shape=(x.shape[1],))
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


def test_loss_adv_vae():
    loss = loss_adv_vae(x, y, model, w_model=1., w_recon=0.)
    loss_with_recon = loss_adv_vae(x, y, model, w_model=1., w_recon=1.)
    assert loss > 0.
    assert loss_with_recon > loss
