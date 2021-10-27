import numpy as np
from alibi_detect.cd import FETDrift

n, n_features = 100, 1


def test_fetdrift():
    # Reference data
    np.random.seed(0)
    p_h0 = 0.5
    p_h1 = 0.3
    x_ref = np.random.choice(2, n, p=[1 - p_h0, p_h0])

    # Instantiate detector
    cd = FETDrift(x_ref=x_ref, p_val=0.05)

    # Test predict
    x = np.random.choice(2, n, p=[1 - p_h1, p_h1])
    cd.predict(x)
