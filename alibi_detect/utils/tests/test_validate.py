from alibi_detect.utils.loading import validate_config
from alibi_detect.version import __version__, __config_spec__
from pydantic import ValidationError
import pytest
import numpy as np

# Define a detector config dict
mmd_cfg = {
    'meta': {
        'version': __version__,
        'config_spec': __config_spec__,
    },
    'name': 'MMDDrift',
    'x_ref': np.array([[-0.30074928], [1.50240758], [0.43135768], [2.11295779], [0.79684913]]),
    'p_val': 0.05
}
cfgs = [mmd_cfg]
n_tests = len(cfgs)


@pytest.fixture
def select_cfg(request):
    return cfgs[request.param]


@pytest.mark.parametrize('select_cfg', list(range(n_tests)), indirect=True)
def test_validate_config(select_cfg):
    cfg = select_cfg

    # Original cfg
    # Check original cfg doesn't raise errors
    print(cfg)
    cfg_full = validate_config(cfg, resolved=True)
    print(cfg_full)

    # Check cfg is returned with meta==None
    assert cfg_full.pop('meta', True) is None

    # Check remaining values of items in cfg unchanged
    for k, v in cfg.items():
        assert np.all((v == cfg_full[k]))  # use np.all to deal with x_ref comparision

    # Check original cfg doesn't raise errors in the unresolved case
    cfg_unres = cfg.copy()
    cfg_unres['x_ref'] = 'x_ref.npy'
    _ = validate_config(cfg_unres)
    print(cfg_unres)
    assert not cfg.get('meta').get('version_warning')

    # Check warning raised and warning field added if version or config_spec different
    cfg_err = cfg.copy()
    cfg_err['version'] = '0.1.x'
    with pytest.warns(Warning):  # error will be raised if a warning IS NOT raised
        cfg_err = validate_config(cfg_err, resolved=True)
    assert cfg_err.get('meta').get('version_warning')

    cfg_err = cfg.copy()
    cfg_err['config_spec'] = '0.x'
    with pytest.warns(Warning):  # error will be raised if a warning IS NOT raised
        cfg_err = validate_config(cfg_err, resolved=True)
    assert cfg_err.get('meta').get('version_warning')

    # Check ValueError raised if name unrecognised
    cfg_err = cfg.copy()
    cfg_err['name'] = 'MMDDriftWrong'
    with pytest.raises(ValueError):
        cfg_err = validate_config(cfg_err, resolved=True)
    assert not cfg_err.get('meta').get('version_warning')

    # Check ValidationError raised if unrecognised field or type wrong
    cfg_err = cfg.copy()
    cfg_err['p_val'] = [cfg['p_val']]  # p_val should be float not list
    with pytest.raises(ValidationError):
        cfg_err = validate_config(cfg_err, resolved=True)
    assert not cfg_err.get('meta').get('version_warning')

    cfg_err = cfg.copy()
    cfg_err['wrong_var'] = 42.0
    with pytest.raises(ValidationError):
        cfg_err = validate_config(cfg_err, resolved=True)
    assert not cfg_err.get('meta').get('version_warning')
