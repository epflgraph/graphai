from xgboost import XGBRegressor

from definitions import FUNDING_DIR
from utils.text.io import mkdir, read_json, save_json


def save_features(features, attributes, name):
    # Create directory if it does not exist
    dirname = f'{FUNDING_DIR}/models/features-{name}'
    mkdir(dirname)

    # Save model features and attributes
    save_json(features, f'{dirname}/features.json')
    save_json(attributes, f'{dirname}/attributes.json')


def load_features(name):
    dirname = f'{FUNDING_DIR}/models/features-{name}'

    # Read features and attributes
    features = read_json(f'{dirname}/features.json')
    attributes = read_json(f'{dirname}/attributes.json')

    return features, attributes


def save_xgb_params(xgb_params, evaluation_summary, features_name, name):
    # Create directory if it does not exist
    dirname = f'{FUNDING_DIR}/models/features-{features_name}/xgb-params-{name}'
    mkdir(dirname)

    # Save xgb_params and evaluation_summary
    save_json(xgb_params, f'{dirname}/xgb_params.json')
    save_json(evaluation_summary, f'{dirname}/evaluation_summary.json')


def load_xgb_params(features_name, name):
    dirname = f'{FUNDING_DIR}/models/features-{features_name}/xgb-params-{name}'

    # Read xgb_params and evaluation_summary
    xgb_params = read_json(f'{dirname}/xgb_params.json')
    evaluation_summary = read_json(f'{dirname}/evaluation_summary.json')

    return xgb_params, evaluation_summary


def save_model(model, features_name, xgb_params_name):
    # Create directory if it does not exist
    dirname = f'{FUNDING_DIR}/models/features-{features_name}/xgb-params-{xgb_params_name}'
    mkdir(dirname)

    # Save model
    model.save_model(f'{dirname}/model.json')


def load_model(features_name, xgb_params_name):
    dirname = f'{FUNDING_DIR}/models/features-{features_name}/xgb-params-{xgb_params_name}'

    # Load model
    model = XGBRegressor()
    model.load_model(f'{dirname}/model.json')

    return model