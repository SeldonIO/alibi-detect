# TODO pytorch
import os
import textwrap
from typing import Union
from pathlib import Path
import logging
import nlp
import numpy as np
from transformers import AutoTokenizer
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.utils.schemas import DETECTOR_CONFIGS
from alibi_detect.utils.saving import save_model, save_tokenizer, save_config
from pydantic import ValidationError

logger = logging.getLogger(__name__)

EXAMPLES = ['imdb_mmd']


def gen_example_config(example: str, filepath: Union[str, os.PathLike], verbose: bool = False):
    """
    Generate a detector's config file and all related artefacts for a given example. The config file and artefacts
    are saved to disk. Running `load_detector(filename)` will instantiate the detector.

    Parameters
    ----------
    example
        The example to fetch e.g. `'imdb_mmd'`.
    filepath
        The filepath to save the config.toml file and its artefacts to.
    verbose
        Whether to print progress updates.
    """
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    if example.lower() == 'imdb_mmd':
        _gen_imdb_mmd(filepath, verbose)
    else:
        raise ValueError('The requested `example` is not recognised. Choose from %s' % str(EXAMPLES))


def _gen_imdb_mmd(filepath: Path, verbose: bool = False):
    """
    Generate config for the imdb_mmd example. Drift detection on the imdb dataset with an MMDDrift detector.
    Code is taken (with minor modification) from examples/cd_text_imdb.ipynb.

    Parameters
    ----------
    filepath
        The filepath to save the config.toml file and its artefacts to.
    """
    # Misc settings
    max_len = 100  # for preprocess_drift
    batch_size = 32  # for preprocess_drift
    n_sample = 1000
    backend = 'tensorflow'
    model_name = 'bert-base-cased'

    # Load and save reference data
    if verbose:
        print('Downloading IMDB dataset and saving reference data')
    data = nlp.load_dataset('imdb')
    X, y = [], []
    for x in data['train']:
        X.append(x['text'])
        y.append(x['label'])
    X = np.array(X)
    y = np.array(y)
    X_ref = _random_sample(X, y, proba_zero=.5, n=n_sample)[0]
    x_ref_path = str(filepath.joinpath('x_ref.npy'))
    np.save(x_ref_path, X_ref)

    # Tokenizer
    if verbose:
        print('Loading and saving tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(list(X_ref), pad_to_max_length=True,
                       max_length=max_len, return_tensors='tf')
    cfg_token = save_tokenizer(tokenizer, filepath)

    # Embedding
    if verbose:
        print('Loading and saving tokenizer')
    emb_type = 'hidden_state'
    n_layers = 8
    layers = [-_ for _ in range(1, n_layers + 1)]
    embedding = TransformerEmbedding(model_name, emb_type, layers)

    # Untrained AutoEncoder Model
    if verbose:
        print('Creating and saving Untrained AutoEncoder Model')
    x_emb = embedding(tokens)
    from alibi_detect.cd.tensorflow import UAE
    enc_dim = 32
    shape = (x_emb.shape[1],)
    model = UAE(input_layer=embedding, shape=shape, enc_dim=enc_dim)
    cfg_model, cfg_embed = save_model(model, filepath, (enc_dim,), backend)

    # Set preprocess_fn config
    cfg_preprocess = {
        'src': '@cd.%s.preprocess.preprocess_drift' % backend,
        'batch_size': batch_size,
        'max_len': max_len,
        'tokenizer': cfg_token,
        'embedding': cfg_embed,
        'model': cfg_model
    }

    # Get detector config and populate
    if verbose:
        print('Creating and saving detector config.toml')
    detector_name = 'MMDDrift'
    cfg = {
        'name': detector_name,
        'x_ref': x_ref_path,
        'backend': backend,
        'preprocess_fn': cfg_preprocess,
    }
    cfg = save_config(cfg, filepath)

    # Validate config
    try:
        DETECTOR_CONFIGS[detector_name](**cfg).dict()
    except ValidationError as e:
        raise Exception('There appears to be an issue with the generated detector config.') from e

    # Finish
    if verbose:
        print(textwrap.dedent('''
        Finished. The detector can be instantiated by running:
        
        from alibi_detect.utils.saving import load_detector
        detector = load_detector('%s')
        ''' % filepath))


def _random_sample(X: np.ndarray, y: np.ndarray, proba_zero: float, n: int):
    # proba_zero = fraction with label 0 (=negative sentiment)
    if len(y.shape) == 1:
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
    else:
        idx_0 = np.where(y[:, 0] == 1)[0]
        idx_1 = np.where(y[:, 1] == 1)[0]
    n_0, n_1 = int(n * proba_zero), int(n * (1 - proba_zero))
    idx_0_out = np.random.choice(idx_0, n_0, replace=False)
    idx_1_out = np.random.choice(idx_1, n_1, replace=False)
    X_out = np.concatenate([X[idx_0_out], X[idx_1_out]])
    y_out = np.concatenate([y[idx_0_out], y[idx_1_out]])
    return X_out, y_out
