"""
This script is an example of using `jupytext` to execute notebooks for testing instead of relying on `nbmake`
plugin. This approach may be more flexible if our requirements change in the future.
"""

import glob
from pathlib import Path
import pytest
from jupytext.cli import jupytext

# Set of all example notebooks
# NOTE: we specifically get only the name of the notebook not the full path as we want to
# use these as variables on the command line for `pytest` for the workflow executing only
# changed notebooks. `pytest` does not allow `/` as part of the test name for the -k argument.
# This also means that the approach is limited to all notebooks being in the `NOTEBOOK_DIR`
# top-level path.
NOTEBOOK_DIR = 'doc/source/examples'
ALL_NOTEBOOKS = {Path(x).name for x in glob.glob(str(Path(NOTEBOOK_DIR).joinpath('*.ipynb')))}

# The following set includes notebooks which are not to be executed during notebook tests.
# These are typically those that would take too long to run in a CI environment or impractical
# due to other dependencies (e.g. downloading large datasets
EXCLUDE_NOTEBOOKS = {
    # the following are all long-running
    'cd_distillation_cifar10.ipynb',
    'cd_ks_cifar10.ipynb',
    'cd_mmd_cifar10.ipynb',
    'od_llr_genome.ipynb',
    'od_llr_mnist.ipynb',
    'od_seq2seq_synth.ipynb',
    # the following requires a k8s cluster
    'alibi_detect_deploy.ipynb',
    # the following require downloading large datasets
    'cd_online_camelyon.ipynb',
    'cd_text_amazon.ipynb',
    # the following require complex dependencies
    'cd_mol.ipynb',  # complex to install pytorch-geometric
}

EXECUTE_NOTEBOOKS = ALL_NOTEBOOKS - EXCLUDE_NOTEBOOKS


@pytest.mark.timeout(600)
@pytest.mark.parametrize("notebook", EXECUTE_NOTEBOOKS)
def test_notebook_execution(notebook):
    notebook = Path(NOTEBOOK_DIR, notebook)
    jupytext(args=[str(notebook), "--execute"])
