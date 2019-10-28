import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_accuracy_metrics(y_true, y_pred):
    """ Get the Average Absolute Percent Error (AAPE) and Median Absolute Percent Error (MAPE)
    :param y_true:
    :param y_pred:
    :return:
    """
    errors = np.absolute(y_true - y_pred) / y_true

    metrics = dict()
    metrics['aape'] = np.round(np.mean(errors), 4)
    metrics['mape'] = np.round(np.median(errors), 4)

    return metrics


def plot_error_distribution(y_true, y_pred):
    errors = np.absolute(y_true - y_pred) / y_true
    aape = np.round(np.mean(errors), 4)
    mape = np.round(np.median(errors), 4)

    fig, ax = plt.subplots()
    fig.set_size_inches((8, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    errors.hist(bins=100, ax=ax, color='gray')
    plt.axvline(aape, color='r', linestyle='dashed', linewidth=1, label='aape')
    plt.axvline(mape, color='b', linestyle='dashed', linewidth=1, label='mape')
    ax.legend(frameon=False)
    # plt.show()

    return ax


def get_feature_importance_plot(tree_model, features):
    importace_scores = tree_model.feature_importances_

    s = pd.Series(importace_scores, index=features)
    s = s.sort_values(ascending=False)

    fig, ax = plt.subplots()
    fig.set_size_inches((12, 4))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    s.plot(kind='bar', ax=ax, color='gray')
    fig.savefig('report/figures/feature_importance_GBDT.svg', format='svg', dpi=1200)
    fig.savefig('report/figures/feature_importance_GBDT.pdf', format='pdf', dpi=1200)
    # plt.show()

    return ax


