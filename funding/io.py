from tsfresh.feature_extraction.settings import from_columns

from xgboost import XGBRegressor

from definitions import FUNDING_DIR
from utils.text.io import mkdir, read_json, save_json


def save_model(model, X, name):
    # Create directory for model if it does not exist
    model_dirname = f'{FUNDING_DIR}/models/{name}'
    mkdir(model_dirname)

    # Save model features
    save_json(from_columns(X), f'{model_dirname}/features.json')

    # Save model
    model.save_model(f'{model_dirname}/model.json')


def save_scores(min_year, max_year, scores, name):
    model_dirname = f'{FUNDING_DIR}/models/{name}'

    # Save data
    data = {
        'min_year': min_year,
        'max_year': max_year,
        'scores': scores
    }
    save_json(data, f'{model_dirname}/scores.json')


def load_model(name):
    model_dirname = f'{FUNDING_DIR}/models/{name}'

    # Load model
    model = XGBRegressor()
    model.load_model(f'{model_dirname}/model.json')

    return model


def load_features(name):
    model_dirname = f'{FUNDING_DIR}/models/{name}'

    # Read features
    features = read_json(f'{model_dirname}/features.json')

    return features
