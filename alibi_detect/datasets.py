import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from typing import Tuple, Union
from alibi_detect.utils.data import Bunch

pd.options.mode.chained_assignment = None  # default='warn'


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
