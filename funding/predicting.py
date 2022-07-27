import pandas as pd

from funding.data_processing import build_data
from funding.io import load_features, load_model

from utils.text.io import log


def predict(model, X):
    # Predict values using model
    y = model.predict(X)

    # Convert numpy.ndarray to pd.Series
    y = pd.Series(y, index=X.index)

    return y


def predict_concepts_year(year, concept_ids, features_name, xgb_params_name, debug=False):
    # Load features
    log(f'Loading features from disk...', debug)
    features, _ = load_features(features_name)

    # Load xgb_params
    log(f'Loading model from disk...', debug)
    model = load_model(features_name, xgb_params_name)

    min_year = year - 3
    max_year = year - 1

    # Build data
    log(f'Building data for features {features_name}...', debug)
    X = build_data(min_year=min_year, max_year=max_year, concept_ids=concept_ids, features_name=features_name, split_y=False, debug=False)
    log(X, debug)

    # Predict values for given year
    log(f'Predicting values...', debug)
    y = predict(model, X)
    log(y, debug)

    return y
