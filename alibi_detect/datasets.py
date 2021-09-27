import dill
import io
from io import BytesIO
import logging
import numpy as np
import os
import pandas as pd
import requests
from requests import RequestException
from scipy.io import arff
from sklearn.datasets import fetch_kddcup99
from typing import List, Tuple, Union
from xml.etree import ElementTree
from alibi_detect.utils.data import Bunch

# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

pd.options.mode.chained_assignment = None  # default='warn'

logger = logging.getLogger(__name__)


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
    data_raw = fetch_kddcup99(subset=None, data_home=None, percent10=percent10)

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


def load_url_arff(url: str, dtype: Union[str, np.dtype] = np.float32) -> np.ndarray:
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
        resp = requests.get(url, timeout=2)
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
        url_corruption = os.path.join(url, corr + '.npy')
        try:
            resp = requests.get(url_corruption, timeout=2)
            resp.raise_for_status()
        except RequestException:
            logger.exception("Could not connect, URL may be out of service")
            raise
        X_corr = np.load(BytesIO(resp.content))[istart:iend].astype('float32')
        X[i * n:(i + 1) * n] = X_corr

    # get labels
    url_labels = os.path.join(url, 'labels.npy')
    try:
        resp = requests.get(url_labels, timeout=2)
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
        resp = requests.get(url, timeout=2)
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
    path_attack = os.path.join(url, dataset, 'attacks', model, attack)
    path_data = path_attack + '.npz'
    path_meta = path_attack + '_meta.pickle'
    # get adversarial instances and labels
    try:
        resp = requests.get(path_data, timeout=2)
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
        resp = requests.get(path_meta, timeout=2)
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
        resp = requests.get(url_labels, timeout=2)
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
        resp = requests.get(url_labels, timeout=2)
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
    path_data = os.path.join(url, fold + '.npz')
    try:
        resp = requests.get(path_data, timeout=2)
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
        resp = requests.get('https://storage.googleapis.com/seldon-datasets/genome/label_dict.json', timeout=2)
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
