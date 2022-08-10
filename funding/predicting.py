import pandas as pd

from funding.data_processing import build_data, time_period_first_date, date_time_period
from funding.io import load_model

from utils.text.io import log


def predict(model, X):
    # Predict values using model
    y = model.predict(X)

    # Convert numpy.ndarray to pd.Series
    y = pd.Series(y, index=X.index)

    return y


def predict_concepts_time_period(time_period, concept_ids, features_name, xgb_params_name, debug=False):
    # Check time period is in the correct format
    year, quarter = time_period.split('-Q')
    year = int(year)
    quarter = int(quarter)
    assert year > 0 and quarter in [1, 2, 3, 4], f'Incorrect format for time_period. Should be <year>-Q<quarter>, e.g. 2025-Q3.'

    # Load xgb_params
    log(f'Loading model from disk...', debug)
    model = load_model(features_name, xgb_params_name)

    max_time_period = time_period
    min_time_period = f'1990-Q1'

    min_date = time_period_first_date(min_time_period)
    max_date = time_period_first_date(max_time_period)

    # Build data for prediction
    log(f'Building data for features {features_name}...', debug)
    df, X = build_data(min_date=min_date, max_date=max_date, concept_ids=concept_ids, features_name=features_name, split_y=False, return_df=True, debug=False)
    log(X, debug)

    # Predict values for given time period
    log(f'Predicting values...', debug)
    y = predict(model, X)
    log(y, debug)

    return df, y
