import pandas as pd

from interfaces.db import DB
from funding.preprocessing import build_time_series
from funding.training import extract_features, select_features, train_regressor

pd.set_option('display.width', 320)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)


if __name__ == '__main__':
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
