import numpy as np
import pandas as pd

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table


def sq(x):
    return x ** 2


def err():
    return np.random.normal() / 10


def load_data(n, length, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = list(range(length))

    df = []
    y = []
    for i in range(n):
        for x_0 in x:
            value = sq(x_0) * (1 + err())
            df.append([i, x_0, value])
        y.append(sq(length) * (1 + err()))

    df = pd.DataFrame(df, columns=['id', 'time', 'value'])
    y = pd.Series(y)

    return df, y


if __name__ == '__main__':
    # Create mock data
    df, y = load_data(100, 100, seed=0)

    print('Shape of df:', df.shape)
    print('Shape of y', y.shape)

    # Feature extraction
    X = extract_features(df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
    print('Shape of df after feature extraction: ', X.shape)

    # Feature selection
    X_filtered = select_features(X, y)
    print('Shape of df after feature selection: ', X_filtered.shape)

    # Relevance table
    rt = calculate_relevance_table(X, y)

    pd.set_option('display.width', 320)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 10)
    print(rt)
