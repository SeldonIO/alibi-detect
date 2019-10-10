import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_outlier_scores(scores: np.ndarray,
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
    df = pd.DataFrame(dict(idx=np.arange(len(scores)), score=scores, label=target))
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.idx, group.score, marker='o', linestyle='', ms=6, label=labels[name])
    plt.plot(np.arange(len(scores)), np.ones(len(scores)) * threshold, color='g', label='Threshold')
    plt.xlabel('Number of Instances')
    plt.ylabel('Outlier Score')
    ax.legend()
    plt.show()
