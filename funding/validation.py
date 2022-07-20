import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, RepeatedKFold

from xgboost import XGBRegressor

from funding.preprocessing import build_time_series, split_last_year
from funding.training import extract_features
from funding.io import load_features, save_scores

from interfaces.db import DB
from utils.text.io import log


def evaluate(X, y, xgb_params=None, debug=False):
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    if xgb_params is None:
        xgb_params = {}

    # Create model
    model = XGBRegressor(**xgb_params)

    # Create cross-validation object
    n_splits = 10
    n_repeats = 3
    cv = RepeatedKFold(n_splits=10, n_repeats=3)

    # Execute cross-validation and compute cv score
    fit_params = {
        'verbose': False,
        'eval_set': [(X_test, y_test)]
    }
    cv_results = cross_validate(model, X_train, y_train, cv=cv, return_estimator=True, n_jobs=-1, fit_params=fit_params)
    avg_n_trees = int(np.round(np.mean([estimator.best_ntree_limit for estimator in cv_results['estimator']])))
    log(f'Optimal number of trees, averaged over all cv folds and rounded: {avg_n_trees}', debug)

    cv_score = cv_results['test_score'].mean()
    log(f'CV SCORE [R2 score (1 - (residual sum squares)/(total sum squares))] after {n_splits}x{n_repeats} cross-validation: {cv_score}', debug)

    # Train model with the full train data
    model.fit(X_train, y_train, **fit_params)

    # Performance of model on train data
    train_score = model.score(X_train, y_train)
    log(f'TRAIN SCORE [R2 score (1 - (residual sum squares)/(total sum squares))]: {train_score}', debug)

    # Performance of model on test data
    test_score = model.score(X_test, y_test)
    log(f'TEST SCORE [R2 score (1 - (residual sum squares)/(total sum squares))]: {test_score}', debug)

    return {
        'train': train_score,
        'cv': cv_score,
        'test': test_score,
        'diff-train-cv': abs(train_score - cv_score),
        'diff-train-test': abs(train_score - test_score),
        'diff-test-cv': abs(test_score - cv_score),
        'avg_n_trees': avg_n_trees
    }


def evaluate_model(min_year, max_year, concept_ids=None, name='', xgb_params=None, debug=False):

    assert min_year < max_year, f'min_year ({min_year}) should be lower than max_year ({max_year})'

    if concept_ids is None:
        db = DB()
        concept_ids = db.get_crunchbase_concept_ids()

    # Create time series with data from database
    log(f'Creating time series for time window {min_year}-{max_year}...', debug)
    df = build_time_series(min_year, max_year, concept_ids=concept_ids, debug=False)

    # Split df rows into < max_year (training data) and = max_year (response variable)
    df, y = split_last_year(df, max_year)

    # Load features
    log(f'Loading model features from disk...', debug)
    features = load_features(name)

    # Extract model features
    log(f'Extracting model features...', debug)
    X = extract_features(df, features)

    # Evaluate model
    log(f'Evaluating model...', debug)
    scores = evaluate(X, y, xgb_params)

    # Save results
    save_scores(min_year, max_year, scores, name)
