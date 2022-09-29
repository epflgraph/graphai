from xgboost import XGBRegressor

from funding.data_processing import build_data
from funding.io import load_features, load_xgb_params, save_model

from utils.text.io import log


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


def create_model(features_name, xgb_params_name, debug=False):
    # Load features
    log(f'Loading features from disk...', debug)
    features, _ = load_features(features_name)

    # Load xgb_params
    log(f'Loading xgb_params from disk...', debug)
    xgb_params, _ = load_xgb_params(features_name, xgb_params_name)

    # Build data
    log(f'Building data for features {features_name}...', debug)
    X, y = build_data(features_name=features_name, split_y=True, debug=False)

    # Train model
    log(f'Training model...', debug)
    model = train_model(X, y, xgb_params=xgb_params)

    log(f'Saving model to disk...', debug)
    save_model(model, features_name, xgb_params_name)
