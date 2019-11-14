import numpy as np
import pandas as pd
from typing import Tuple, Union


class Bunch(dict):
    """
    Container object for internal datasets
    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def sample_df(df: pd.DataFrame,
              n: int):
    """ Sample n instances from the dataframe df. """
    if n < df.shape[0]+1:
        replace = False
    else:
        replace = True
    return df.sample(n=n, replace=replace)


def create_outlier_batch(data: np.ndarray,
                         target: np.ndarray,
                         n_samples: int,
                         perc_outlier: int) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """ Create a batch with a defined percentage of outliers. """

    # create df
    data = pd.DataFrame(data=data)
    data['target'] = target

    # separate inlier and outlier data
    normal = data[data['target'] == 0]
    outlier = data[data['target'] == 1]

    if n_samples == 1:
        n_outlier = np.random.binomial(1, .01 * perc_outlier)
        n_normal = 1 - n_outlier
    else:
        n_outlier = int(perc_outlier * .01 * n_samples)
        n_normal = int((100 - perc_outlier) * .01 * n_samples)

    # draw samples
    batch_normal = sample_df(normal, n_normal)
    batch_outlier = sample_df(outlier, n_outlier)

    batch = pd.concat([batch_normal, batch_outlier])
    batch = batch.sample(frac=1).reset_index(drop=True)

    is_outlier = batch['target'].values
    batch.drop(columns=['target'], inplace=True)

    return Bunch(data=batch.values, target=is_outlier, target_names=['normal', 'outlier'])
