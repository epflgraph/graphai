import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from interfaces.db import DB
from funding.predict import build_time_series

pd.set_option('display.width', 320)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)

# x^2
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

    df = pd.DataFrame(df, columns=['concept_id', 'year', 'amount'])
    y = pd.Series(y)

    return df, y


def extract_features(df):
    # Feature extraction
    X = extract_features(df, column_id='concept_id', column_sort='year', column_value='amount',
                         default_fc_parameters=ComprehensiveFCParameters(),
                         impute_function=impute)  # we impute = remove all NaN features automatically

    print('Shape of df after feature extraction: ', X.shape)

    return X


def select_features(X, y, manual=False):
    # Feature selection
    X_filtered = select_features(X, y)
    print('Shape of df after feature selection: ', X_filtered.shape)

    if manual:
        # Manual feature selection
        rt = calculate_relevance_table(X, y)
        rt = rt[rt['p_value'] <= 0.05]
        selected_features = list(rt['feature'])
        X_filtered = X[selected_features]
        print('Shape of df after manual feature selection: ', X_filtered.shape)

    return X_filtered


def train_regressor(X, y):
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Model training: TODO fine-tune hyperparameters to reduce overfitting
    model = XGBRegressor()

    # Cross-validation to evaluate performance
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
    print(f'Average R2 score (1 - (residual sum squares)/(total sum squares)) after 3x10 cross-validation: {scores.mean()}')

    # Train model with full data
    model.fit(X_train, y_train)

    # Feature importances
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    print('Feature importances:')
    print(feature_importances.head(10))

    # Performance of model on train data
    score = model.score(X_train, y_train)
    print(f'R2 score (1 - (residual sum squares)/(total sum squares)) on train data: {score}')

    y_pred = model.predict(X_train)

    score = model.score(X_train, y_pred)
    print(f'R2 score (1 - (residual sum squares)/(total sum squares)) on train data vs. pred responses: {score}')

    # Performance of model on test data
    score = model.score(X_test, y_test)
    print(f'R2 score (1 - (residual sum squares)/(total sum squares)) on test data: {score}')

    y_pred = model.predict(X_test)

    score = model.score(X_test, y_pred)
    print(f'R2 score (1 - (residual sum squares)/(total sum squares)) on test data vs. pred responses: {score}')


if __name__ == '__main__':
    # # Create mock data
    # df, y = load_data(100, 100, seed=0)
    # print('Shape of df:', df.shape)
    # print('Shape of y', y.shape)
    #
    # # Extract features and flatten time series
    # X = extract_features_and_flatten_ts(df, y)
    #
    # # Train regressor and evaluate performance
    # train_regressor(X, y)

    db = DB()
    concept_ids = db.get_crunchbase_concept_ids()
    df = build_time_series(2018, 2021, concept_ids=concept_ids, debug=False)

    # Extract response variable
    y = df[df['year'] == 2021]
    df = df[df['year'] != 2021]

    # Set concept_id as index
    y.index = y['concept_id']
    y = y['amount']

    # Extract features and select most relevant ones
    X = extract_features(df)
    X = select_features(X, y)

    # Train regressor and evaluate performance
    train_regressor(X, y)
