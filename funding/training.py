import pandas as pd

import tsfresh as tsf
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

from xgboost import XGBRegressor


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


def cross_validate(X, y):
    # Create model
    model = XGBRegressor()

    # Create cross-validation object
    n_splits = 10
    n_repeats = 3
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)

    # Execute cross-validation and keep scores
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    mean_score = scores.mean()
    print(f'Average R2 score (1 - (residual sum squares)/(total sum squares)) after {n_splits}x{n_repeats} cross-validation: {mean_score}')

    return mean_score


def get_feature_importances(X, model):
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    return feature_importances


def train_regressor(X, y):
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Model training: TODO fine-tune hyperparameters to reduce overfitting
    model = XGBRegressor()

    # Train model
    model.fit(X_train, y_train)

    # Feature importances
    feature_importances = get_feature_importances(X_train, model)
    print('Feature importances:')
    print(feature_importances.head(10))

    # Performance of model on train data
    score = model.score(X_train, y_train)
    print(f'R2 score (1 - (residual sum squares)/(total sum squares)) on train data: {score}')

    y_pred = model.predict(X_train)

    score = model.score(X_train, y_pred)
    print(f'R2 score (1 - (residual sum squares)/(total sum squares)) on train data vs. pred responses: {score}')

    # Performance of model on test data
    score = model.score(X_test, y_test)
    print(f'R2 score (1 - (residual sum squares)/(total sum squares)) on test data: {score}')

    y_pred = model.predict(X_test)

    score = model.score(X_test, y_pred)
    print(f'R2 score (1 - (residual sum squares)/(total sum squares)) on test data vs. pred responses: {score}')

    return model
