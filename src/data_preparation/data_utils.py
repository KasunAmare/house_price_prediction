import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    df = pd.read_csv(file_path)
    # num_recs, num_dims = df.shape
    # print('num records: ', num_recs)
    # print('num dimensions: ', num_dims)

    return df


def get_features_with_missing_values(data):
    features = data.columns
    null_features = features[data.isnull().any()]
    null_features = list(null_features)

    # dictionary with key: feature, value: number of rows with values missing
    missing_features = dict()

    for f in null_features:
        temp = data[f]
        missing_features[f] = temp[temp.isnull()].shape[0]

    return missing_features


def replace_missing_with_single_value(data, feature, value, add_feature=False):
    """
    :param data: data with missing values, pandas dataframe
    :param feature: feature with missing values
    :param value: single value to replace
    :param add_feature: add an additional feature to indicate replaced value
    :return:
    """

    if add_feature:
        msk = (data[feature].isnull())
        s = pd.Series(0, index=data.index)
        s[msk] = 1
        data[feature + '_missing_replaced'] = s

    data.loc[data[feature].isnull(), feature] = value

    return data


def get_value_distribution(data, feature):
    """ Inspect the unique value distribution
    :param data: data all
    :param feature: feature to inspect
    :return: a data frame with unique values and the frequency of appearance
    """
    tpl = np.unique(data[feature], return_counts=True)
    df = pd.DataFrame(list(tpl)).T
    df.columns = ['value', 'count']

    return df


def three_sigma_outlier(data, feature):
    """ Remove rows where the feature value is greater than 3 standard deviations from the mean
    :param data: data
    :param feature: feature to base outliers on
    :return: outliers removed data
    """
    ave = data[feature].mean()
    std = data[feature].std()

    interval_upper = ave + 3 * std

    outlier_rows = data[data[feature] > interval_upper].index

    data = data.drop(outlier_rows, axis=0)

    return data


def plot_distribution(data, feature):
    fig, ax = plt.subplots()
    fig.set_size_inches((8, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mean = data[feature].mean()
    std = data[feature].std()

    data[feature].hist(bins=500, ax=ax, color='gray')
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='mean')
    plt.axvline(mean + 3 * std, color='r', linestyle='dashed', linewidth=1, label=' mean + 3 * sigma')

    plt.xlim(0, 4000000)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    ax.legend(frameon=False)

    ax.grid(b=None)

    return fig, ax


def split_input_output_data(data, output_column):
    y = data[output_column]

    x = data.drop(columns=[output_column])

    return x, y




