import numpy as np
from alibi_detect.cd import CVMDrift

n, n_features = 100, 1


def test_cvmdrift():
    # Reference data
    np.random.seed(0)
    x_ref = np.random.randn(*(n, n_features))

    # Instantiate detector
    cd = CVMDrift(x_ref=x_ref, p_val=0.05)

    # Test predict
    np.random.seed(1)
    x = np.random.randn(*(n, n_features))
    cd.predict(x)
