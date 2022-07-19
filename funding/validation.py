from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

from xgboost import XGBRegressor

from funding.preprocessing import build_time_series, split_last_year
from funding.training import extract_features
from funding.io import load_features, save_scores

from interfaces.db import DB
from utils.text.io import log


def evaluate(X, y):
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

    return {
        'train': train_score,
        'cv': cv_score,
        'test': test_score
    }


def evaluate_model(min_year, max_year, concept_ids=None, name='', debug=False):

    assert min_year < max_year, f'min_year ({min_year}) should be lower than max_year ({max_year})'

    if concept_ids is None:
        db = DB()
        concept_ids = db.get_crunchbase_concept_ids()

    # Create time series with data from database
    log(f'Creating time series for time window {min_year}-{max_year}...', debug)
    df = build_time_series(min_year, max_year, concept_ids=concept_ids, debug=False)

    # Split df rows into < max_year (training data) and = max_year (response variable)
    df, y = split_last_year(df, max_year)

    # Load model
    log(f'Loading model and features from disk...', debug)
    features = load_features(name)

    # Extract features and select most relevant ones
    log(f'Extracting model features...', debug)
    X = extract_features(df, features)

    # Evaluate model
    log(f'Evaluating model...', debug)
    scores = evaluate(X, y)

    # Save results
    save_scores(min_year, max_year, scores, name)
