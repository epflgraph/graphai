import pandas as pd

from tsfresh.feature_extraction.settings import from_columns

from funding.preprocessing import build_time_series, split_last_year
from funding.training import extract_features, select_features, train_regressor

from definitions import FUNDING_DIR
from interfaces.db import DB
from utils.text.io import mkdir, save_json

pd.set_option('display.width', 320)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)


if __name__ == '__main__':
    min_year = 2018
    max_year = 2021

    db = DB()
    concept_ids = db.get_crunchbase_concept_ids()

    # Create time series with data from database
    df = build_time_series(min_year, max_year, concept_ids=concept_ids, debug=False)

    # Split df rows into < max_year (training data) and = max_year (response variable)
    df, y = split_last_year(df, max_year)

    # Extract features and select most relevant ones
    X = extract_features(df)
    X = select_features(X, y)

    # Train regressor and evaluate performance
    model = train_regressor(X, y)

    # Save model and its properties
    name = 'test'
    model_dirname = f'{FUNDING_DIR}/models/{name}'
    mkdir(model_dirname)
    save_json(from_columns(X), f'{model_dirname}/features.json')
    model.save_model(f'{model_dirname}/model.json')
