import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict


def plot_instance_outlier(od_preds: Dict,
                          target: np.ndarray,
                          labels: np.ndarray,
                          threshold: float) -> None:
    """
    Scatter plot of a batch of outlier scores compared to the outlier threshold.

    Parameters
    ----------
    scores
        Outlier scores.
    target
        Ground truth.
    labels
        List with names of classification labels.
    threshold
        Threshold used to classify outliers.
    """
    scores = od_preds['data']['instance_score']
    df = pd.DataFrame(dict(idx=np.arange(len(scores)), score=scores, label=target))
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.idx, group.score, marker='o', linestyle='', ms=6, label=labels[name])
    plt.plot(np.arange(len(scores)), np.ones(len(scores)) * threshold, color='g', label='Threshold')
    plt.xlabel('Number of Instances')
    plt.ylabel('Instance Level Outlier Score')
    ax.legend()
    plt.show()


def plot_feature_outlier_image(od_preds: Dict,
                               X: np.ndarray,
                               X_recon: np.ndarray = None,
                               max_outliers: int = 5,
                               outliers_only: bool = False,
                               n_channels: int = 3,
                               figsize: tuple = (20, 20)) -> None:
    """
    Plot feature (pixel) wise outlier scores for images.

    Parameters
    ----------
    od_preds
        Output of an outlier detector's prediction.
    X
        Batch of instances to apply outlier detection to.
    X_recon
        Reconstructed instances of X.
    max_outliers
        Maximum number of outliers to display.
    outliers_only
        Whether to only show outliers or not.
    n_channels
        Number of channels of the images.
    figsize
        Tuple for the figure size.
    """
    scores = od_preds['data']['feature_score']
    if outliers_only:
        outlier_ids = list(np.where(od_preds['data']['is_outlier'])[0])
    else:
        outlier_ids = list(range(len(od_preds['data']['is_outlier'])))
    n_outliers = min(max_outliers, len(outlier_ids))
    n_cols = 2

    if n_channels == 3:
        n_cols += 2

    if X_recon is not None:
        n_cols += 1

    fig, axes = plt.subplots(nrows=n_outliers, ncols=n_cols, figsize=figsize)

    n_subplot = 1
    for i in range(n_outliers):

        idx = outlier_ids[i]

        X_outlier = X[idx]
        plt.subplot(n_outliers, n_cols, n_subplot)
        plt.axis('off')
        if i == 0:
            plt.title('Original')
        plt.imshow(X_outlier)
        n_subplot += 1

        if X_recon is not None:
            plt.subplot(n_outliers, n_cols, n_subplot)
            plt.axis('off')
            if i == 0:
                plt.title('Reconstruction')
            plt.imshow(X_recon[idx])
            n_subplot += 1

        plt.subplot(n_outliers, n_cols, n_subplot)
        plt.axis('off')
        if i == 0:
            plt.title('Outlier Score Channel 0')
        plt.imshow(scores[idx][:, :, 0])
        n_subplot += 1

        if n_channels == 3:
            plt.subplot(n_outliers, n_cols, n_subplot)
            plt.axis('off')
            if i == 0:
                plt.title('Outlier Score Channel 1')
            plt.imshow(scores[idx][:, :, 1])
            n_subplot += 1

            plt.subplot(n_outliers, n_cols, n_subplot)
            plt.axis('off')
            if i == 0:
                plt.title('Outlier Score Channel 2')
            plt.imshow(scores[idx][:, :, 2])
            n_subplot += 1

    plt.show()


def plot_feature_outlier_tabular():
    pass


def plot_feature_outlier_ts():
    pass
