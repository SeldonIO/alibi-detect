import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from odcd.utils.data import Bunch

pd.options.mode.chained_assignment = None  # default='warn'


def fetch_kdd(target=['dos', 'r2l', 'u2r', 'probe'],
              keep_cols=['srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                         'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                         'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                         'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                         'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                         'dst_host_srv_rerror_rate'],
              percent10=True):
    """ KDD Cup '99 dataset. Detect computer network intrusions. """

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

    return Bunch(data=data.values, target=is_outlier, target_names=['is_outlier'])
