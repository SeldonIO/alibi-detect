import numpy as np
import pytest
from pydantic import ValidationError

from alibi_detect.saving import validate_config
from alibi_detect.saving.saving import X_REF_FILENAME
from alibi_detect.version import __config_spec__, __version__
from copy import deepcopy

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

# Define a detector config dict without meta (as simple as it gets!)
mmd_cfg_nometa = deepcopy(mmd_cfg)
mmd_cfg_nometa.pop('meta')


@pytest.mark.parametrize('cfg', [mmd_cfg])
def test_validate_config(cfg):
    # Original cfg
    # Check original cfg doesn't raise errors
    cfg_full = validate_config(cfg, resolved=True)

    # Check cfg is returned with correct metadata
    meta = cfg_full.get('meta')  # pop as don't want to compare meta to cfg in next bit
    assert meta['version'] == __version__
    assert meta['config_spec'] == __config_spec__
    assert not meta.pop('version_warning')  # pop this one to remove from next check

    # Check remaining values of items in cfg unchanged
    for k, v in cfg.items():
        assert np.all((v == cfg_full[k]))  # use np.all to deal with x_ref comparision

    # Check original cfg doesn't raise errors in the unresolved case
    cfg_unres = cfg.copy()
    cfg_unres['x_ref'] = X_REF_FILENAME
    _ = validate_config(cfg_unres)
    assert not cfg.get('meta').get('version_warning')

    # Check warning raised and warning field added if version or config_spec different
    cfg_err = cfg.copy()
    cfg_err['meta']['version'] = '0.1.x'
    with pytest.warns(Warning):  # error will be raised if a warning IS NOT raised
        cfg_err = validate_config(cfg_err, resolved=True)
    assert cfg_err.get('meta').get('version_warning')

    cfg_err = cfg.copy()
    cfg_err['meta']['config_spec'] = '0.x'
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


@pytest.mark.parametrize('cfg', [mmd_cfg_nometa])
def test_validate_config_wo_meta(cfg):
    # Check a config w/o a meta dict can be validated
    _ = validate_config(cfg, resolved=True)

    # Check the unresolved case
    cfg_unres = cfg.copy()
    cfg_unres['x_ref'] = X_REF_FILENAME
    _ = validate_config(cfg_unres)
