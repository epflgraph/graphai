import pandas as pd

import tsfresh as tsf
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

from xgboost import XGBRegressor

from funding.preprocessing import build_time_series, split_last_year
from funding.io import save_model

from interfaces.db import DB
from utils.text.io import log
from utils.time.date import now


def extract_features(df, features=None):
    if features is None:
        features = {'amount': ComprehensiveFCParameters()}

    X = tsf.extract_features(df, column_id='concept_id', column_sort='year', column_value='amount',
                         kind_to_fc_parameters=features,
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


def get_feature_importances(X, model):
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    return feature_importances


def train_model(X, y, xgb_params=None):
    if xgb_params is None:
        xgb_params = {}

    model = XGBRegressor(**xgb_params)

    # Train model
    model.fit(X, y)

    # # Feature importances
    # feature_importances = get_feature_importances(X, model)
    # print('Feature importances:')
    # print(feature_importances.head(10))

    return model


def create_model(min_year, max_year, concept_ids=None, name='', xgb_params=None, debug=False):

    assert min_year < max_year, f'min_year ({min_year}) should be lower than max_year ({max_year})'

    if concept_ids is None:
        db = DB()
        concept_ids = db.get_crunchbase_concept_ids()

    # Create time series with data from database
    log(f'Creating time series for time window {min_year}-{max_year}...', debug)
    df = build_time_series(min_year, max_year, concept_ids=concept_ids, debug=False)

    # Split df rows into < max_year (training data) and = max_year (response variable)
    df, y = split_last_year(df, max_year)

    # Extract features and select most relevant ones
    log(f'Extracting features and selecting the most relevant ones...', debug)
    X = extract_features(df)
    X = select_features(X, y)

    # Train model and evaluate performance
    log(f'Training model...', debug)
    model = train_model(X, y, xgb_params=xgb_params)

    # Save model and its features
    if not name:
        name = now().strftime('%Y%m%d%H%M%S')
    log(f'Saving model to disk under the name "{name}"...', debug)
    save_model(model, X, name=name)
