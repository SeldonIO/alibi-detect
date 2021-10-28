import numpy as np
import pytest
import scipy
from packaging import version
if version.parse(scipy.__version__) >= version.parse('1.7.0'):
    from alibi_detect.cd import CVMDrift

n, n_features = 100, 1


@pytest.mark.skipif(version.parse(scipy.__version__) < version.parse('1.7.0'),
                    reason="Requires scipy version >= 1.7.0")
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
