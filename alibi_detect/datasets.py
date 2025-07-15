import io
import logging
from io import BytesIO
from typing import List, Tuple, Type, Union, Optional, Dict
from xml.etree import ElementTree

import dill
import numpy as np
import pandas as pd
import requests
from alibi_detect.utils.data import Bunch
from alibi_detect.utils.url import _join_url
from requests import RequestException
from urllib.error import URLError
from scipy.io import arff
from sklearn.datasets import fetch_kddcup99

# Financial data imports
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import fredapi
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

pd.options.mode.chained_assignment = None  # default='warn'

logger = logging.getLogger(__name__)

"""Number of seconds to wait for URL requests before raising an error."""
TIMEOUT = 10


def fetch_kdd(target: list = ['dos', 'r2l', 'u2r', 'probe'],
              keep_cols: list = ['srv_count', 'serror_rate', 'srv_serror_rate',
                                 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                                 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                                 'dst_host_srv_count', 'dst_host_same_srv_rate',
                                 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                                 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                                 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                                 'dst_host_srv_rerror_rate'],
              percent10: bool = True,
              return_X_y: bool = False) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    KDD Cup '99 dataset. Detect computer network intrusions.

    Parameters
    ----------
    target
        List with attack types to detect.
    keep_cols
        List with columns to keep. Defaults to continuous features.
    percent10
        Bool, whether to only return 10% of the data.
    return_X_y
        Bool, whether to only return the data and target values or a Bunch object.

    Returns
    -------
    Bunch
        Dataset and outlier labels (0 means 'normal' and 1 means 'outlier').
    (data, target)
        Tuple if 'return_X_y' equals True.
    """

    # fetch raw data
    try:
        data_raw = fetch_kddcup99(subset=None, data_home=None, percent10=percent10)
    except URLError:
        logger.exception("Could not connect, URL may be out of service")
        raise

    # specify columns
    cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    # create dataframe
    data = pd.DataFrame(data=data_raw['data'], columns=cols)

    # add target to dataframe
    data['attack_type'] = data_raw['target']

    # specify and map attack types
    attack_list = np.unique(data['attack_type'])
    attack_category = ['dos', 'u2r', 'r2l', 'r2l', 'r2l', 'probe', 'dos', 'u2r',
                       'r2l', 'dos', 'probe', 'normal', 'u2r', 'r2l', 'dos', 'probe',
                       'u2r', 'probe', 'dos', 'r2l', 'dos', 'r2l', 'r2l']

    attack_types = {}
    for i, j in zip(attack_list, attack_category):
        attack_types[i] = j

    data['attack_category'] = 'normal'
    for k, v in attack_types.items():
        data['attack_category'][data['attack_type'] == k] = v

    # define target
    data['target'] = 0
    for t in target:
        data['target'][data['attack_category'] == t] = 1
    is_outlier = data['target'].values

    # define columns to be dropped
    drop_cols = []
    for col in data.columns.values:
        if col not in keep_cols:
            drop_cols.append(col)

    if drop_cols != []:
        data.drop(columns=drop_cols, inplace=True)

    if return_X_y:
        return data.values, is_outlier

    return Bunch(data=data.values,
                 target=is_outlier,
                 target_names=['normal', 'outlier'],
                 feature_names=keep_cols)


def load_url_arff(url: str, dtype: Type[np.generic] = np.float32) -> np.ndarray:
    """
    Load arff files from url.

    Parameters
    ----------
    url
        Address of arff file.

    Returns
    -------
    Arrays with data and labels.
    """
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise
    data = arff.loadarff(io.StringIO(resp.text))[0]
    return np.array(data.tolist(), dtype=dtype)


def fetch_ecg(return_X_y: bool = False) \
        -> Union[Bunch, Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Fetch ECG5000 data. The dataset contains 5000 ECG's, originally obtained from
    Physionet (https://archive.physionet.org/cgi-bin/atm/ATM) under the name
    "BIDMC Congestive Heart Failure Database(chfdb)", record "chf07".

    Parameters
    ----------
    return_X_y
        Bool, whether to only return the data and target values or a Bunch object.

    Returns
    -------
    Bunch
        Train and test datasets with labels.
    (train data, train target), (test data, test target)
        Tuple of tuples if 'return_X_y' equals True.
    """
    Xy_train = load_url_arff('https://storage.googleapis.com/seldon-datasets/ecg/ECG5000_TRAIN.arff')
    X_train, y_train = Xy_train[:, :-1], Xy_train[:, -1]
    Xy_test = load_url_arff('https://storage.googleapis.com/seldon-datasets/ecg/ECG5000_TEST.arff')
    X_test, y_test = Xy_test[:, :-1], Xy_test[:, -1]
    if return_X_y:
        return (X_train, y_train), (X_test, y_test)
    else:
        return Bunch(data_train=X_train,
                     data_test=X_test,
                     target_train=y_train,
                     target_test=y_test)


def fetch_cifar10c(corruption: Union[str, List[str]], severity: int, return_X_y: bool = False) \
        -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    Fetch CIFAR-10-C data. Originally obtained from https://zenodo.org/record/2535967#.XkKh2XX7Qts and
    introduced in "Hendrycks, D and Dietterich, T.G. Benchmarking Neural Network Robustness to Common Corruptions
    and Perturbations. In 7th International Conference on Learning Represenations, 2019.".

    Parameters
    ----------
    corruption
        Corruption type. Options can be checked with `get_corruption_cifar10c()`.
        Alternatively, specify 'all' for all corruptions at a severity level.
    severity
        Severity level of corruption (1-5).
    return_X_y
        Bool, whether to only return the data and target values or a Bunch object.

    Returns
    -------
    Bunch
        Corrupted dataset with labels.
    (corrupted data, target)
        Tuple if 'return_X_y' equals True.
    """
    url = 'https://storage.googleapis.com/seldon-datasets/cifar10c/'
    n = 10000  # instances per corrupted test set
    istart, iend = (severity - 1) * n, severity * n  # idx for the relevant severity level
    corruption_list = corruption_types_cifar10c()  # get all possible corruption types
    # convert input to list
    if isinstance(corruption, str) and corruption != 'all':
        corruption = [corruption]
    elif corruption == 'all':
        corruption = corruption_list
    for corr in corruption:  # check values in corruptions
        if corr not in corruption_list:
            raise ValueError(f'{corr} is not a valid corruption type.')
    # get corrupted data
    shape = ((len(corruption)) * n, 32, 32, 3)
    X = np.zeros(shape)
    for i, corr in enumerate(corruption):
        url_corruption = _join_url(url, corr + '.npy')
        try:
            resp = requests.get(url_corruption, timeout=TIMEOUT)
            resp.raise_for_status()
        except RequestException:
            logger.exception("Could not connect, URL may be out of service")
            raise
        X_corr = np.load(BytesIO(resp.content))[istart:iend].astype('float32')
        X[i * n:(i + 1) * n] = X_corr

    # get labels
    url_labels = _join_url(url, 'labels.npy')
    try:
        resp = requests.get(url_labels, timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise
    y = np.load(BytesIO(resp.content))[istart:iend].astype('int64')
    if X.shape[0] != y.shape[0]:
        repeat = X.shape[0] // y.shape[0]
        y = np.tile(y, (repeat,))

    if return_X_y:
        return (X, y)
    else:
        return Bunch(data=X, target=y)


def google_bucket_list(url: str, folder: str, filetype: str = None, full_path: bool = False) -> List[str]:
    """
    Retrieve list with items in google bucket folder.

    Parameters
    ----------
    url
        Bucket directory.
    folder
        Folder to retrieve list of items from.
    filetype
        File extension, e.g. `npy` for saved numpy arrays.

    Returns
    -------
    List with items in the folder of the google bucket.
    """
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise
    root = ElementTree.fromstring(resp.content)
    bucket_list = []
    for r in root:
        if list(r):
            filepath = r[0].text
            if filetype is not None:
                if filepath.startswith(folder) and filepath.endswith(filetype):
                    istart, istop = filepath.find('/') + 1, filepath.find('.')
                    bucket_list.append(filepath[istart:istop])
            else:
                if filepath.startswith(folder):
                    istart, istop = filepath.find('/') + 1, filepath.find('.')
                    bucket_list.append(filepath[istart:istop])
    return bucket_list


def corruption_types_cifar10c() -> List[str]:
    """
    Retrieve list with corruption types used in CIFAR-10-C.

    Returns
    -------
    List with corruption types.
    """
    url = 'https://storage.googleapis.com/seldon-datasets/'
    folder = 'cifar10c'
    filetype = 'npy'
    corruption_types = google_bucket_list(url, folder, filetype)
    corruption_types.remove('labels')
    return corruption_types


def fetch_attack(dataset: str, model: str, attack: str, return_X_y: bool = False) \
        -> Union[Bunch, Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Load adversarial instances for a given dataset, model and attack type.

    Parameters
    ----------
    dataset
        Dataset under attack.
    model
        Model under attack.
    attack
        Attack name.
    return_X_y
        Bool, whether to only return the data and target values or a Bunch object.

    Returns
    -------
    Bunch
        Adversarial instances with original labels.
    (train data, train target), (test data, test target)
        Tuple of tuples if 'return_X_y' equals True.
    """
    # define paths
    url = 'https://storage.googleapis.com/seldon-datasets/'
    path_attack = _join_url(url, [dataset, 'attacks', model, attack])
    path_data = path_attack + '.npz'
    path_meta = path_attack + '_meta.pickle'
    # get adversarial instances and labels
    try:
        resp = requests.get(path_data, timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise
    data = np.load(BytesIO(resp.content))
    X_train, X_test = data['X_train_adv'], data['X_test_adv']
    y_train, y_test = data['y_train'], data['y_test']

    if return_X_y:
        return (X_train, y_train), (X_test, y_test)

    # get metadata
    try:
        resp = requests.get(path_meta, timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise
    meta = dill.load(BytesIO(resp.content))
    return Bunch(data_train=X_train,
                 data_test=X_test,
                 target_train=y_train,
                 target_test=y_test,
                 meta=meta)


def fetch_nab(ts: str,
              return_X_y: bool = False
              ) -> Union[Bunch, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Get time series in a DataFrame from the Numenta Anomaly Benchmark: https://github.com/numenta/NAB.

    Parameters
    ----------
    ts

    return_X_y
        Bool, whether to only return the data and target values or a Bunch object.

    Returns
    -------
    Bunch
        Dataset and outlier labels (0 means 'normal' and 1 means 'outlier') in DataFrames with timestamps.
    (data, target)
        Tuple if 'return_X_y' equals True.
    """
    url_labels = 'https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json'
    try:
        resp = requests.get(url_labels, timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise
    labels_json = resp.json()
    outliers = labels_json[ts + '.csv']
    if not outliers:
        logger.warning('The dataset does not contain any outliers.')
    url = 'https://raw.githubusercontent.com/numenta/NAB/master/data/' + ts + '.csv'
    df = pd.read_csv(url, header=0, index_col=0)
    labels = np.zeros(df.shape[0])
    for outlier in outliers:
        outlier_id = np.where(df.index == outlier)[0][0]
        labels[outlier_id] = 1
    df.index = pd.to_datetime(df.index)
    df_labels = pd.DataFrame(data={'is_outlier': labels}, index=df.index)

    if return_X_y:
        return df, df_labels

    return Bunch(data=df,
                 target=df_labels,
                 target_names=['normal', 'outlier'])


def get_list_nab() -> list:
    """
    Get list of possible time series to retrieve from the Numenta Anomaly Benchmark: https://github.com/numenta/NAB.

    Returns
    -------
    List with time series names.
    """
    url_labels = 'https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json'
    try:
        resp = requests.get(url_labels, timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise
    labels_json = resp.json()
    files = [k[:-4] for k, v in labels_json.items()]
    return files


def load_genome_npz(fold: str, return_labels: bool = False) \
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    url = 'https://storage.googleapis.com/seldon-datasets/genome/'
    path_data = _join_url(url, fold + '.npz')
    try:
        resp = requests.get(path_data, timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise
    data = np.load(BytesIO(resp.content))
    if return_labels:
        return data['x'], data['is_outlier'], data['y']
    else:
        return data['x'], data['is_outlier']


def fetch_genome(return_X_y: bool = False, return_labels: bool = False) -> Union[Bunch, tuple]:
    """
    Load genome data including their labels and whether they are outliers or not. More details about the data can be
    found in the readme on https://console.cloud.google.com/storage/browser/seldon-datasets/genome/.
    The original data can be found here: https://drive.google.com/drive/folders/1Ht9xmzyYPbDouUTl_KQdLTJQYX2CuclR.

    Parameters
    ----------
    return_X_y
        Bool, whether to only return the data and target values or a Bunch object.
    return_labels
        Whether to return the genome labels which are detailed in the `label_json` dict
        of the returned Bunch object.

    Returns
    -------
    Bunch
        Training, validation and test data, whether they are outliers and optionally including the
        genome labels which are specified in the `label_json` key as a dictionary.
    (data, outlier) or (data, outlier, target)
        Tuple for the train, validation and test set with either the data and whether they
        are outliers or the data, outlier flag and labels for the genomes if 'return_X_y' equals True.
    """
    data_train = load_genome_npz('train_in', return_labels=return_labels)
    data_val_in = load_genome_npz('val_in', return_labels=return_labels)
    data_val_ood = load_genome_npz('val_ood', return_labels=return_labels)
    data_val = (
        np.concatenate([data_val_in[0], data_val_ood[0]]),
        np.concatenate([data_val_in[1], data_val_ood[1]])
    )
    data_test_in = load_genome_npz('test_in', return_labels=return_labels)
    data_test_ood = load_genome_npz('test_ood', return_labels=return_labels)
    data_test = (
        np.concatenate([data_test_in[0], data_test_ood[0]]),
        np.concatenate([data_test_in[1], data_test_ood[1]])
    )
    if return_labels:
        data_val += (np.concatenate([data_val_in[2], data_val_ood[2]]),)  # type: ignore
        data_test += (np.concatenate([data_test_in[2], data_test_ood[2]]),)  # type: ignore
    if return_X_y:
        return data_train, data_val, data_test
    try:
        resp = requests.get('https://storage.googleapis.com/seldon-datasets/genome/label_dict.json', timeout=TIMEOUT)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise

    label_dict = resp.json()
    bunch = Bunch(
        data_train=data_train[0],
        data_val=data_val[0],
        data_test=data_test[0],
        outlier_train=data_train[1],
        outlier_val=data_val[1],
        outlier_test=data_test[1],
        label_dict=label_dict
    )
    if not return_labels:
        return bunch
    else:
        bunch['target_train'] = data_train[2]  # type: ignore
        bunch['target_val'] = data_val[2]  # type: ignore
        bunch['target_test'] = data_test[2]  # type: ignore
        return bunch


# ============================================================================
# FINANCIAL DATA FUNCTIONS
# ============================================================================

def get_financial_crisis_presets() -> Dict[str, Dict]:
    """
    Get predefined financial crisis configurations for drift detection studies.

    Returns
    -------
    Dict
        Dictionary of crisis configurations with start/end dates and descriptions.
    """
    return {
        '2008_financial_crisis': {
            'description': '2008 Global Financial Crisis (Subprime mortgage crisis)',
            'pre_crisis_start': '2007-01-01',
            'pre_crisis_end': '2008-07-31',
            'crisis_start': '2008-09-01',
            'crisis_end': '2009-04-30',
            'typical_tickers': ['SPY', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'QQQ', 'IWM'],
            'description_long': ('Period covering the subprime mortgage crisis, '
                                 'Lehman Brothers collapse, and subsequent market turmoil')
        },
        '2020_covid_crisis': {
            'description': '2020 COVID-19 Market Crash',
            'pre_crisis_start': '2019-01-01',
            'pre_crisis_end': '2020-02-14',
            'crisis_start': '2020-02-20',
            'crisis_end': '2020-05-31',
            'typical_tickers': ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI'],
            'description_long': 'Period covering the COVID-19 pandemic market crash and initial recovery'
        },
        '2000_dotcom_crash': {
            'description': '2000 Dot-com Bubble Burst',
            'pre_crisis_start': '1999-01-01',
            'pre_crisis_end': '2000-03-10',
            'crisis_start': '2000-03-11',
            'crisis_end': '2002-10-09',
            'typical_tickers': ['SPY', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV'],
            'description_long': 'Period covering the dot-com bubble burst and subsequent tech stock collapse'
        },
        '2011_european_debt': {
            'description': '2011 European Debt Crisis',
            'pre_crisis_start': '2010-01-01',
            'pre_crisis_end': '2011-07-31',
            'crisis_start': '2011-08-01',
            'crisis_end': '2012-06-30',
            'typical_tickers': ['SPY', 'XLF', 'EFA', 'VGK', 'XLE', 'XLK'],
            'description_long': 'Period covering European sovereign debt crisis and eurozone instability'
        }
    }


def fetch_financial_crisis(crisis: str = '2008_financial_crisis',
                           tickers: Optional[List[str]] = None,
                           data_source: str = 'yfinance',
                           fred_api_key: Optional[str] = None,
                           include_macro: bool = False,
                           return_X_y: bool = False,
                           return_raw: bool = False,
                           min_history: int = 100) -> Union[Bunch, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fetch financial crisis data for drift detection analysis.

    This function downloads historical financial data for pre-crisis and crisis periods,
    providing clean datasets suitable for distribution drift analysis, particularly
    correlation structure changes during market stress.

    Parameters
    ----------
    crisis
        Crisis identifier. Options: '2008_financial_crisis', '2020_covid_crisis',
        '2000_dotcom_crash', '2011_european_debt', or custom dates as dict.
    tickers
        List of ticker symbols to download. If None, uses typical tickers for the crisis.
    data_source
        Data source: 'yfinance' (default) or 'fred' for economic indicators.
    fred_api_key
        FRED API key if using FRED data source or including macro indicators.
    include_macro
        Whether to include macroeconomic indicators from FRED.
    return_X_y
        If True, return (pre_crisis_returns, crisis_returns) tuple.
    return_raw
        If True, return raw price data instead of returns.
    min_history
        Minimum number of trading days required for each ticker.

    Returns
    -------
    Bunch
        Financial crisis dataset with pre-crisis and crisis period data.
        - data_pre: Pre-crisis period returns/prices
        - data_crisis: Crisis period returns/prices
        - tickers: List of successful ticker symbols
        - dates_pre: Date range for pre-crisis period
        - dates_crisis: Date range for crisis period
        - crisis_info: Metadata about the crisis
    (pre_crisis_data, crisis_data)
        Tuple if return_X_y=True.

    Examples
    --------
    >>> # Load 2008 financial crisis data
    >>> data = fetch_financial_crisis('2008_financial_crisis')
    >>> pre_returns = data.data_pre
    >>> crisis_returns = data.data_crisis
    >>> print(f"Pre-crisis shape: {pre_returns.shape}")
    >>> print(f"Crisis shape: {crisis_returns.shape}")

    >>> # Load with custom tickers
    >>> custom_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    >>> data = fetch_financial_crisis('2020_covid_crisis', tickers=custom_tickers)

    >>> # Get raw price data instead of returns
    >>> prices = fetch_financial_crisis('2008_financial_crisis', return_raw=True)

    >>> # Include macroeconomic data (requires FRED API key)
    >>> data = fetch_financial_crisis('2008_financial_crisis',
    ...                              include_macro=True,
    ...                              fred_api_key='your_api_key')
    """

    # Get crisis configuration
    crisis_presets = get_financial_crisis_presets()

    if isinstance(crisis, str):
        if crisis not in crisis_presets:
            available = ', '.join(crisis_presets.keys())
            raise ValueError(f"Unknown crisis '{crisis}'. Available: {available}")
        crisis_config = crisis_presets[crisis]
    elif isinstance(crisis, dict):
        required_keys = ['pre_crisis_start', 'pre_crisis_end', 'crisis_start', 'crisis_end']
        if not all(key in crisis for key in required_keys):
            raise ValueError(f"Custom crisis dict must contain: {required_keys}")
        crisis_config = crisis
    else:
        raise ValueError("Crisis must be string identifier or dict with date ranges")

    # Set default tickers if none provided
    if tickers is None:
        tickers = crisis_config.get('typical_tickers',
                                    ['SPY', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'QQQ', 'IWM'])

    logger.info(f"Fetching financial crisis data: {crisis_config.get('description', crisis)}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Pre-crisis: {crisis_config['pre_crisis_start']} to {crisis_config['pre_crisis_end']}")
    logger.info(f"Crisis: {crisis_config['crisis_start']} to {crisis_config['crisis_end']}")

    # Download financial data
    if data_source == 'yfinance':
        pre_data, crisis_data, successful_tickers = _fetch_yfinance_data(
            tickers, crisis_config, min_history
        )
    elif data_source == 'fred':
        if not HAS_FRED:
            raise ImportError("fredapi package required for FRED data. Install with: pip install fredapi")
        if fred_api_key is None:
            raise ValueError("fred_api_key required when using FRED data source")
        pre_data, crisis_data, successful_tickers = _fetch_fred_data(
            tickers, crisis_config, fred_api_key, min_history
        )
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    # Add macroeconomic indicators if requested
    if include_macro:
        if not HAS_FRED:
            raise ImportError("fredapi package required for macro data. Install with: pip install fredapi")
        if fred_api_key is None:
            raise ValueError("fred_api_key required for macroeconomic data")

        macro_pre, macro_crisis = _fetch_macro_indicators(
            crisis_config, fred_api_key, pre_data.index, crisis_data.index
        )

        # Combine financial and macro data
        pre_data = pd.concat([pre_data, macro_pre], axis=1)
        crisis_data = pd.concat([crisis_data, macro_crisis], axis=1)
        successful_tickers.extend(macro_pre.columns.tolist())

    # Convert to returns if not returning raw data
    if not return_raw:
        pre_returns = pre_data.pct_change().dropna()
        crisis_returns = crisis_data.pct_change().dropna()
    else:
        pre_returns = pre_data
        crisis_returns = crisis_data

    # Validate data quality
    if len(pre_returns) < min_history or len(crisis_returns) < min_history // 2:
        logger.warning(f"Limited data available: pre={len(pre_returns)}, crisis={len(crisis_returns)}")

    logger.info(f"Successfully loaded data for {len(successful_tickers)} assets")
    logger.info(f"Pre-crisis: {pre_returns.shape}, Crisis: {crisis_returns.shape}")

    if return_X_y:
        return pre_returns, crisis_returns

    return Bunch(
        data_pre=pre_returns,
        data_crisis=crisis_returns,
        tickers=successful_tickers,
        dates_pre=(crisis_config['pre_crisis_start'], crisis_config['pre_crisis_end']),
        dates_crisis=(crisis_config['crisis_start'], crisis_config['crisis_end']),
        crisis_info=crisis_config,
        feature_names=successful_tickers,
        target_names=['pre_crisis', 'crisis'],
        description=f"Financial crisis dataset: {crisis_config.get('description', 'Custom crisis')}"
    )


def _fetch_yfinance_data(tickers: List[str],
                         crisis_config: Dict,
                         min_history: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Fetch data using yfinance.
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance package required. Install with: pip install yfinance")

    pre_data = {}
    crisis_data = {}
    successful_tickers = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)

            # Download pre-crisis data
            pre_hist = stock.history(
                start=crisis_config['pre_crisis_start'],
                end=crisis_config['pre_crisis_end']
            )

            # Download crisis data
            crisis_hist = stock.history(
                start=crisis_config['crisis_start'],
                end=crisis_config['crisis_end']
            )

            # Validate data quality
            if len(pre_hist) >= min_history and len(crisis_hist) >= min_history // 2:
                pre_data[ticker] = pre_hist['Close']
                crisis_data[ticker] = crisis_hist['Close']
                successful_tickers.append(ticker)
                logger.debug(f"✅ {ticker}: {len(pre_hist)} + {len(crisis_hist)} days")
            else:
                logger.warning(f"❌ {ticker}: Insufficient data ({len(pre_hist)} + {len(crisis_hist)} days)")

        except Exception as e:
            logger.warning(f"❌ {ticker}: Download failed - {e}")
            continue

    if len(successful_tickers) == 0:
        raise ValueError("No valid tickers could be downloaded")

    # Create DataFrames and align dates
    pre_df = pd.DataFrame(pre_data).dropna()
    crisis_df = pd.DataFrame(crisis_data).dropna()

    return pre_df, crisis_df, successful_tickers


def _fetch_fred_data(tickers: List[str],
                     crisis_config: Dict,
                     fred_api_key: str,
                     min_history: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Fetch economic data using FRED API.
    """
    fred = fredapi.Fred(api_key=fred_api_key)

    pre_data = {}
    crisis_data = {}
    successful_tickers = []

    for series_id in tickers:
        try:
            # Download full series
            data = fred.get_series(
                series_id,
                start=crisis_config['pre_crisis_start'],
                end=crisis_config['crisis_end']
            )

            # Split into pre-crisis and crisis periods
            pre_end = pd.to_datetime(crisis_config['pre_crisis_end'])
            crisis_start = pd.to_datetime(crisis_config['crisis_start'])

            pre_series = data[data.index <= pre_end]
            crisis_series = data[data.index >= crisis_start]

            # Validate data quality
            if len(pre_series) >= min_history // 10 and len(crisis_series) >= min_history // 20:
                pre_data[series_id] = pre_series
                crisis_data[series_id] = crisis_series
                successful_tickers.append(series_id)
                logger.debug(f"✅ {series_id}: {len(pre_series)} + {len(crisis_series)} observations")
            else:
                logger.warning(f"❌ {series_id}: Insufficient data")

        except Exception as e:
            logger.warning(f"❌ {series_id}: Download failed - {e}")
            continue

    if len(successful_tickers) == 0:
        raise ValueError("No valid FRED series could be downloaded")

    # Create DataFrames
    pre_df = pd.DataFrame(pre_data).dropna()
    crisis_df = pd.DataFrame(crisis_data).dropna()

    return pre_df, crisis_df, successful_tickers


def _fetch_macro_indicators(crisis_config: Dict,
                            fred_api_key: str,
                            pre_dates: pd.DatetimeIndex,
                            crisis_dates: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch macroeconomic indicators from FRED.
    """
    fred = fredapi.Fred(api_key=fred_api_key)

    # Common macroeconomic indicators
    macro_series = {
        'FEDFUNDS': 'Federal Funds Rate',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'GDP': 'Gross Domestic Product',
        'DEXUSEU': 'USD/EUR Exchange Rate',
        'DGS10': '10-Year Treasury Rate',
        'VIXCLS': 'VIX Volatility Index'
    }

    macro_data = {}

    for series_id, description in macro_series.items():
        try:
            data = fred.get_series(
                series_id,
                start=crisis_config['pre_crisis_start'],
                end=crisis_config['crisis_end']
            )

            if len(data) > 10:  # Minimum data points
                macro_data[series_id] = data
                logger.debug(f"✅ Macro {series_id}: {len(data)} observations")
            else:
                logger.debug(f"❌ Macro {series_id}: Insufficient data")

        except Exception as e:
            logger.debug(f"❌ Macro {series_id}: {e}")
            continue

    if not macro_data:
        logger.warning("No macroeconomic indicators could be loaded")
        return pd.DataFrame(), pd.DataFrame()

    # Create DataFrame and forward-fill missing values
    macro_df = pd.DataFrame(macro_data).fillna(method='ffill')

    # Align with financial data dates
    pre_macro = macro_df.reindex(pre_dates, method='ffill')
    crisis_macro = macro_df.reindex(crisis_dates, method='ffill')

    return pre_macro.dropna(), crisis_macro.dropna()


def create_synthetic_crisis_data(n_assets: int = 8,
                                 n_pre: int = 400,
                                 n_crisis: int = 150,
                                 pre_correlation: float = 0.3,
                                 crisis_correlation: float = 0.6,
                                 volatility_increase: float = 1.5,
                                 random_seed: int = 42,
                                 return_X_y: bool = False) -> Union[Bunch, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create synthetic financial crisis data with controlled correlation changes.

    This function generates realistic financial returns data that exhibits
    the correlation structure changes typical of financial crises, useful
    for testing drift detection methods.

    Parameters
    ----------
    n_assets
        Number of financial assets to simulate.
    n_pre
        Number of pre-crisis observations.
    n_crisis
        Number of crisis observations.
    pre_correlation
        Average correlation during pre-crisis period.
    crisis_correlation
        Average correlation during crisis period.
    volatility_increase
        Factor by which volatility increases during crisis.
    random_seed
        Random seed for reproducibility.
    return_X_y
        If True, return (pre_crisis_data, crisis_data) tuple.

    Returns
    -------
    Bunch
        Synthetic crisis dataset with controlled correlation structure.
    (pre_crisis_data, crisis_data)
        Tuple if return_X_y=True.

    Examples
    --------
    >>> # Create synthetic crisis with moderate correlation increase
    >>> data = create_synthetic_crisis_data(n_assets=10,
    ...                                    pre_correlation=0.25,
    ...                                    crisis_correlation=0.55)
    >>>
    >>> # Test spectral drift detection
    >>> from alibi_detect.cd.spectral import SpectralDrift
    >>> detector = SpectralDrift(data.data_pre.values)
    >>> result = detector.predict(data.data_crisis.values)
    >>> print(f"Spectral ratio: {result['data']['spectral_ratio']:.3f}")
    """

    np.random.seed(random_seed)

    # Asset names
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]

    # Create correlation matrices
    def create_correlation_matrix(base_corr: float, n: int) -> np.ndarray:
        """Create a realistic correlation matrix."""
        # Start with random correlations around base_corr
        corr = np.random.uniform(base_corr - 0.1, base_corr + 0.1, (n, n))

        # Make symmetric
        corr = (corr + corr.T) / 2

        # Set diagonal to 1
        np.fill_diagonal(corr, 1.0)

        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(corr)
        eigenvals = np.maximum(eigenvals, 0.1)  # Minimum eigenvalue
        corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Normalize to correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        return corr

    pre_corr = create_correlation_matrix(pre_correlation, n_assets)
    crisis_corr = create_correlation_matrix(crisis_correlation, n_assets)

    # Base volatility
    base_vol = 0.015  # 1.5% daily volatility

    # Generate returns
    pre_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=pre_corr * (base_vol ** 2),
        size=n_pre
    )

    crisis_returns = np.random.multivariate_normal(
        mean=-np.ones(n_assets) * 0.0005,  # Slight negative drift
        cov=crisis_corr * ((base_vol * volatility_increase) ** 2),
        size=n_crisis
    )

    # Create DataFrames with realistic dates
    pre_dates = pd.date_range(start='2007-01-01', periods=n_pre, freq='B')
    crisis_dates = pd.date_range(start='2008-09-01', periods=n_crisis, freq='B')

    pre_df = pd.DataFrame(pre_returns, index=pre_dates, columns=asset_names)
    crisis_df = pd.DataFrame(crisis_returns, index=crisis_dates, columns=asset_names)

    # Calculate spectral ratio for reference
    pre_eigenvals = np.linalg.eigvals(pre_corr)
    crisis_eigenvals = np.linalg.eigvals(crisis_corr)
    spectral_ratio = np.max(crisis_eigenvals) / np.max(pre_eigenvals)

    logger.info("Synthetic crisis data created:")
    logger.info(f"  Assets: {n_assets}")
    logger.info(f"  Pre-crisis: {n_pre} observations")
    logger.info(f"  Crisis: {n_crisis} observations")
    logger.info(f"  Correlation change: {pre_correlation:.3f} → {crisis_correlation:.3f}")
    logger.info(f"  Spectral ratio: {spectral_ratio:.3f}")

    if return_X_y:
        return pre_df, crisis_df

    return Bunch(
        data_pre=pre_df,
        data_crisis=crisis_df,
        tickers=asset_names,
        correlation_pre=pre_corr,
        correlation_crisis=crisis_corr,
        spectral_ratio=spectral_ratio,
        dates_pre=(str(pre_dates[0].date()), str(pre_dates[-1].date())),
        dates_crisis=(str(crisis_dates[0].date()), str(crisis_dates[-1].date())),
        feature_names=asset_names,
        target_names=['pre_crisis', 'crisis'],
        description=f"Synthetic financial crisis data with {n_assets} assets"
    )


def get_financial_benchmarks() -> Dict[str, Dict]:
    """
    Get predefined financial benchmark datasets for drift detection evaluation.

    Returns
    -------
    Dict
        Dictionary of benchmark configurations for reproducible experiments.
    """
    return {
        'correlation_change_mild': {
            'description': 'Mild correlation structure change',
            'n_assets': 8,
            'n_pre': 400,
            'n_crisis': 150,
            'pre_correlation': 0.25,
            'crisis_correlation': 0.45,
            'volatility_increase': 1.2,
            'expected_spectral_ratio': 1.8
        },
        'correlation_change_moderate': {
            'description': 'Moderate correlation structure change',
            'n_assets': 8,
            'n_pre': 400,
            'n_crisis': 150,
            'pre_correlation': 0.30,
            'crisis_correlation': 0.60,
            'volatility_increase': 1.5,
            'expected_spectral_ratio': 2.0
        },
        'correlation_change_severe': {
            'description': 'Severe correlation structure change',
            'n_assets': 10,
            'n_pre': 500,
            'n_crisis': 200,
            'pre_correlation': 0.20,
            'crisis_correlation': 0.75,
            'volatility_increase': 2.0,
            'expected_spectral_ratio': 3.75
        },
        'high_dimensional': {
            'description': 'High-dimensional financial system',
            'n_assets': 20,
            'n_pre': 300,
            'n_crisis': 100,
            'pre_correlation': 0.15,
            'crisis_correlation': 0.50,
            'volatility_increase': 1.8,
            'expected_spectral_ratio': 3.33
        }
    }


def fetch_financial_benchmark(benchmark: str,
                              random_seed: int = 42,
                              return_X_y: bool = False) -> Union[Bunch, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fetch a predefined financial benchmark dataset.

    Parameters
    ----------
    benchmark
        Benchmark identifier. See get_financial_benchmarks() for options.
    random_seed
        Random seed for reproducibility.
    return_X_y
        If True, return (pre_crisis_data, crisis_data) tuple.

    Returns
    -------
    Bunch or tuple
        Benchmark dataset with known characteristics.

    Examples
    --------
    >>> # Load moderate correlation change benchmark
    >>> data = fetch_financial_benchmark('correlation_change_moderate')
    >>> print(f"Expected spectral ratio: {data.expected_spectral_ratio}")

    >>> # Test with spectral drift detector
    >>> from alibi_detect.cd.spectral import SpectralDrift
    >>> detector = SpectralDrift(data.data_pre.values)
    >>> result = detector.predict(data.data_crisis.values)
    >>> actual_ratio = result['data']['spectral_ratio']
    >>> expected_ratio = data.expected_spectral_ratio
    >>> print(f"Actual ratio: {actual_ratio:.3f}, Expected: {expected_ratio:.3f}")
    """
    benchmarks = get_financial_benchmarks()

    if benchmark not in benchmarks:
        available = ', '.join(benchmarks.keys())
        raise ValueError(f"Unknown benchmark '{benchmark}'. Available: {available}")

    config = benchmarks[benchmark]

    # Create synthetic data with benchmark parameters
    data = create_synthetic_crisis_data(
        n_assets=config['n_assets'],
        n_pre=config['n_pre'],
        n_crisis=config['n_crisis'],
        pre_correlation=config['pre_correlation'],
        crisis_correlation=config['crisis_correlation'],
        volatility_increase=config['volatility_increase'],
        random_seed=random_seed,
        return_X_y=return_X_y
    )

    if return_X_y:
        return data
    
    # At this point, data must be a Bunch, but mypy doesn't know
    # Add type assertion to help mypy
    assert not isinstance(data, tuple), "Expected Bunch when return_X_y=False"
    
    # Add benchmark-specific metadata
    data.benchmark_name = benchmark
    data.expected_spectral_ratio = config['expected_spectral_ratio']
    data.description = f"Financial benchmark: {config['description']}"

    return data


def analyze_financial_data(pre_data: pd.DataFrame,
                           crisis_data: pd.DataFrame,
                           return_full_analysis: bool = False) -> Union[Dict, Bunch]:
    """
    Analyze financial data for distribution drift characteristics.

    Parameters
    ----------
    pre_data
        Pre-crisis financial data (returns or prices).
    crisis_data
        Crisis period financial data (returns or prices).
    return_full_analysis
        If True, return comprehensive analysis including correlations and tests.

    Returns
    -------
    Dict or Bunch
        Analysis results including correlation changes, volatility changes,
        and basic statistical tests.

    Examples
    --------
    >>> data = fetch_financial_crisis('2008_financial_crisis')
    >>> analysis = analyze_financial_data(data.data_pre, data.data_crisis)
    >>> print(f"Spectral ratio: {analysis['spectral_ratio']:.3f}")
    >>> print(f"Correlation change: {analysis['correlation_change']:.3f}")
    """

    # Basic statistics
    pre_corr = pre_data.corr().values
    crisis_corr = crisis_data.corr().values

    # Spectral analysis
    pre_eigenvals = np.linalg.eigvals(pre_corr)
    crisis_eigenvals = np.linalg.eigvals(crisis_corr)
    spectral_ratio = np.max(np.real(crisis_eigenvals)) / np.max(np.real(pre_eigenvals))

    # Correlation changes
    corr_diff = crisis_corr - pre_corr
    correlation_change = np.mean(np.abs(corr_diff[np.triu_indices_from(corr_diff, k=1)]))
    max_correlation_change = np.max(np.abs(corr_diff))

    # Volatility changes
    pre_vol = pre_data.std()
    crisis_vol = crisis_data.std()
    volatility_ratio = np.mean(crisis_vol / pre_vol)

    # Basic analysis results
    analysis = {
        'spectral_ratio': spectral_ratio,
        'correlation_change': correlation_change,
        'max_correlation_change': max_correlation_change,
        'volatility_ratio': volatility_ratio,
        'pre_crisis_shape': pre_data.shape,
        'crisis_shape': crisis_data.shape,
        'n_assets': len(pre_data.columns)
    }

    if return_full_analysis:
        # Extended analysis
        from scipy import stats

        # Statistical tests on means
        mean_tests = {}
        for col in pre_data.columns:
            t_stat, p_val = stats.ttest_ind(pre_data[col], crisis_data[col])
            mean_tests[col] = {'t_statistic': t_stat, 'p_value': p_val}

        # Variance tests
        var_tests = {}
        for col in pre_data.columns:
            f_stat, p_val = stats.levene(pre_data[col], crisis_data[col])
            var_tests[col] = {'f_statistic': f_stat, 'p_value': p_val}

        # Distribution tests (KS test)
        ks_tests = {}
        for col in pre_data.columns:
            ks_stat, p_val = stats.ks_2samp(pre_data[col], crisis_data[col])
            ks_tests[col] = {'ks_statistic': ks_stat, 'p_value': p_val}

        # Return comprehensive Bunch object
        return Bunch(
            **analysis,
            correlation_pre=pre_corr,
            correlation_crisis=crisis_corr,
            correlation_difference=corr_diff,
            eigenvalues_pre=np.real(pre_eigenvals),
            eigenvalues_crisis=np.real(crisis_eigenvals),
            volatility_pre=pre_vol.values,
            volatility_crisis=crisis_vol.values,
            mean_tests=mean_tests,
            variance_tests=var_tests,
            ks_tests=ks_tests
        )

    return analysis


# Convenience function aliases
fetch_crisis_data = fetch_financial_crisis  # Shorter alias
create_crisis_data = create_synthetic_crisis_data  # Shorter alias
