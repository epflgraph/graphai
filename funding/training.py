from xgboost import XGBRegressor

from funding.preprocessing import build_data

from definitions import FUNDING_DIR
from utils.text.io import log, mkdir


def save_model(model, name):
    # Create directory for model if it does not exist
    dirname = f'{FUNDING_DIR}/models/{name}'
    mkdir(dirname)

    # Save model
    model.save_model(f'{dirname}/model.json')


def load_model(name):
    model_dirname = f'{FUNDING_DIR}/models/{name}'

    # Load model
    model = XGBRegressor()
    model.load_model(f'{model_dirname}/model.json')

    return model


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


def create_model(features_name, min_year, max_year, concept_ids=None, xgb_params=None, debug=False):
    # Sanity check
    assert min_year < max_year, f'min_year ({min_year}) should be lower than max_year ({max_year})'

    # Build data
    log(f'Building data for time window {min_year}-{max_year} and features {features_name}...', debug)
    X, y = build_data(min_year, max_year, concept_ids, features_name=features_name, split_y=True, debug=False)

    # Train model
    log(f'Training model...', debug)
    model = train_model(X, y, xgb_params=xgb_params)

    # Save model
    if concept_ids is None:
        name = f'{features_name}-{min_year}-{max_year}-all'
    else:
        name = f'{features_name}-{min_year}-{max_year}-{sum(concept_ids)}'
    log(f'Saving model to disk under the name "{name}"...', debug)
    save_model(model, X, name=name)
