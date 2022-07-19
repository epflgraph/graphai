import pandas as pd

from funding.preprocessing import build_time_series, combine_last_year
from funding.training import extract_features
from funding.io import load_model, load_features

from utils.text.io import log


def predict(model, X):
    # Predict values using model
    y = model.predict(X)

    # Convert numpy.ndarray to pd.Series
    y = pd.Series(y, index=X.index)

    return y


def predict_concepts_year(year, concept_ids, name='', debug=False):
    min_year = year - 3
    max_year = year - 1

    # Create time series with data from database
    log(f'Creating time series for time window {min_year}-{max_year}...', debug)
    df = build_time_series(min_year, max_year, concept_ids=concept_ids, debug=True)
    log(df, debug)

    # Load model and features
    log(f'Loading model and features from disk...', debug)
    model = load_model(name)
    features = load_features(name)

    # Extract model features
    log(f'Extracting model features...', debug)
    X = extract_features(df, features)
    log(X, debug)

    # Predict values for given year
    log(f'Predicting values...', debug)
    y = predict(model, X)
    log(y, debug)

    return combine_last_year(df, y, year)
