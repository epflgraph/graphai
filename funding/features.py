import pandas as pd

import tsfresh as tsf
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

from funding.preprocessing import build_data

from definitions import FUNDING_DIR
from utils.text.io import log, mkdir, read_json, save_json


def save_features(features, attributes, name):
    # Create directory if it does not exist
    dirname = f'{FUNDING_DIR}/models/features-{name}'
    mkdir(dirname)

    # Save model features and attributes
    save_json(features, f'{dirname}/features.json')
    save_json(attributes, f'{dirname}/attributes.json')


def load_features(name):
    dirname = f'{FUNDING_DIR}/models/features-{name}'

    # Read features and attributes
    features = read_json(f'{dirname}/features.json')
    attributes = read_json(f'{dirname}/attributes.json')

    return features, attributes


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


def create_feature_set(min_year, max_year, concept_ids=None, debug=False):
    # Sanity check
    assert min_year < max_year, f'min_year ({min_year}) should be lower than max_year ({max_year})'

    # Build data
    log(f'Building data for time window {min_year}-{max_year} and all features...', debug)
    X, y = build_data(min_year=min_year, max_year=max_year, concept_ids=concept_ids, split_y=True, debug=False)

    # Select most relevant features
    log(f'Extracting features and selecting the most relevant ones...', debug)
    X = select_features(X, y)

    # Save features
    if concept_ids is None:
        name = f'{min_year}-{max_year}-all'
    else:
        name = f'{min_year}-{max_year}-{sum(concept_ids)}'
    log(f'Saving features to disk under the name "{name}"...', debug)
    features = from_columns(X)
    attributes = {
        'min_year': min_year,
        'max_year': max_year,
        'concept_ids': concept_ids
    }
    save_features(features, attributes, name=name)

    return name


