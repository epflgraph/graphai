import pandas as pd

import tsfresh as tsf
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_selection.relevance import calculate_relevance_table

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

from xgboost import XGBRegressor

from definitions import FUNDING_DIR
from utils.text.io import mkdir, save_json


def extract_features(df, kind_to_fc_params=None):
    if kind_to_fc_params is None:
        kind_to_fc_params = {'amount': ComprehensiveFCParameters()}

    X = tsf.extract_features(df, column_id='concept_id', column_sort='year', column_value='amount',
                         kind_to_fc_parameters=kind_to_fc_params,
                         impute_function=impute)  # we impute = remove all NaN features automatically

    print('Shape of df after feature extraction: ', X.shape)

    return X


def select_features(X, y, manual=False):
    # Feature selection
    X_filtered = tsf.select_features(X, y)
    print('Shape of df after feature selection: ', X_filtered.shape)

    if manual:
        # Manual feature selection
        rt = calculate_relevance_table(X, y)
        rt = rt[rt['p_value'] <= 0.05]
        selected_features = list(rt['feature'])
        X_filtered = X[selected_features]
        print('Shape of df after manual feature selection: ', X_filtered.shape)

    return X_filtered


def evaluate_model(X, y):
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Create model
    model = XGBRegressor()

    # Create cross-validation object
    n_splits = 10
    n_repeats = 3
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)

    # Execute cross-validation and compute cv score
    scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
    cv_score = scores.mean()
    print(f'CV SCORE [R2 score (1 - (residual sum squares)/(total sum squares))] after {n_splits}x{n_repeats} cross-validation: {cv_score}')

    # Train model with the full train data
    model.fit(X_train, y_train)

    # Performance of model on train data
    train_score = model.score(X_train, y_train)
    print(f'TRAIN SCORE [R2 score (1 - (residual sum squares)/(total sum squares))]: {train_score}')

    # Performance of model on test data
    test_score = model.score(X_test, y_test)
    print(f'TEST SCORE [R2 score (1 - (residual sum squares)/(total sum squares))]: {test_score}')

    return [('train', train_score), ('cv', cv_score), ('test', test_score)]


def get_feature_importances(X, model):
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    return feature_importances


def train_model(X, y):
    model = XGBRegressor()

    # Train model
    model.fit(X, y)

    # # Feature importances
    # feature_importances = get_feature_importances(X, model)
    # print('Feature importances:')
    # print(feature_importances.head(10))

    return model


def save_model(model, X, name):
    model_dirname = f'{FUNDING_DIR}/models/{name}'
    mkdir(model_dirname)
    save_json(from_columns(X), f'{model_dirname}/features.json')
    model.save_model(f'{model_dirname}/model.json')
