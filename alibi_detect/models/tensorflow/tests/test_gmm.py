import numpy as np
from alibi_detect.models.tensorflow.gmm import gmm_energy, gmm_params

N, K, D = 10, 5, 1
z = np.random.rand(N, D).astype(np.float32)
gamma = np.random.rand(N, K).astype(np.float32)


def test_gmm_params_energy():
    phi, mu, cov, L, log_det_cov = gmm_params(z, gamma)
    assert phi.numpy().shape[0] == K == log_det_cov.shape[0]
    assert mu.numpy().shape == (K, D)
    assert cov.numpy().shape == L.numpy().shape == (K, D, D)
    for _ in range(cov.numpy().shape[0]):
        assert (np.diag(cov[_].numpy()) >= 0.).all()
        assert (np.diag(L[_].numpy()) >= 0.).all()

    sample_energy, cov_diag = gmm_energy(z, phi, mu, cov, L, log_det_cov, return_mean=True)
    assert sample_energy.numpy().shape == cov_diag.numpy().shape == ()

    sample_energy, _ = gmm_energy(z, phi, mu, cov, L, log_det_cov, return_mean=False)
    assert sample_energy.numpy().shape[0] == N
