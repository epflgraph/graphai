from funding.preprocessing import build_time_series, split_last_year
from funding.training import extract_features
from funding.validation import evaluate
from funding.io import load_features

from interfaces.db import DB
from utils.text.io import log


def tune_parameter_single(param_grids, xgb_params, X, y):
    best_score = None
    best_value = None

    param0, grid0 = param_grids[0]

    for value0 in grid0:
        xgb_params[param0] = value0

        cv_score = evaluate(X, y, xgb_params, debug=False)['cv']

        if best_score is None or best_score < cv_score:
            best_score = cv_score
            best_value = value0

        # print(f'{param0}={value0}', f'cv_score={cv_score}')

    # print()
    # print('Best value:')
    # print(f'{param0}={best_value}', f'cv_score={best_score}')

    return best_value


def tune_parameter_pair(param_grids, xgb_params, X, y):
    best_score = None
    best_combination = None

    param0, grid0 = param_grids[0]
    param1, grid1 = param_grids[1]

    for value0 in grid0:
        xgb_params[param0] = value0
        for value1 in grid1:
            xgb_params[param1] = value1

            cv_score = evaluate(X, y, xgb_params, debug=False)['cv']

            if best_score is None or best_score < cv_score:
                best_score = cv_score
                best_combination = (value0, value1)

            # print(f'{param0}={value0}', f'{param1}={value1}', f'cv_score={cv_score}')

    # print()
    # print('Best combination:')
    # print(f'{param0}={best_combination[0]}', f'{param1}={best_combination[1]}', f'cv_score={best_score}')

    return best_combination[0], best_combination[1]


def gen_grid(param, xgb_params, search_spaces, type):
    value = xgb_params[param]

    param_min = search_spaces[param][0]
    param_max = search_spaces[param][1]

    if type == 'int':
        grid = [value + i for i in [-2, -1, 0, 1, 2]]
    elif type == 'float':
        grid = [value + i for i in [-0.2, -0.1, 0, 0.1, 0.2]]
    elif type == 'log':
        grid = [value * i for i in [0.33, 0.6, 1, 2, 4]]
    else:
        grid = []

    grid = [x for x in grid if param_min <= x <= param_max]

    return grid


def tune_all_parameters(xgb_params, search_spaces, X, y):
    # Step 1: Get optimal number of estimators for the given parameters
    evaluation_summary = evaluate(X, y, xgb_params, debug=False)
    xgb_params['n_estimators'] = evaluation_summary['avg_n_trees']

    # Step 2: Tune max_depth and min_child_weight
    md_grid = gen_grid('max_depth', xgb_params, search_spaces, 'int')
    mcw_grid = gen_grid('min_child_weight', xgb_params, search_spaces, 'int')
    max_depth, min_child_weight = tune_parameter_pair([('max_depth', md_grid), ('min_child_weight', mcw_grid)], xgb_params, X, y)
    xgb_params['max_depth'] = max_depth
    xgb_params['min_child_weight'] = min_child_weight

    # Step 3: Tune gamma
    gamma_grid = gen_grid('gamma', xgb_params, search_spaces, 'log')
    gamma = tune_parameter_single([('gamma', gamma_grid)], xgb_params, X, y)
    xgb_params['gamma'] = gamma

    # Step 4: Tune subsample and colsample_bytree
    ss_grid = gen_grid('subsample', xgb_params, search_spaces, 'float')
    cs_grid = gen_grid('colsample_bytree', xgb_params, search_spaces, 'float')
    subsample, colsample_bytree = tune_parameter_pair([('subsample', ss_grid), ('colsample_bytree', cs_grid)], xgb_params, X, y)
    xgb_params['subsample'] = subsample
    xgb_params['colsample_bytree'] = colsample_bytree

    # Step 5: Tune reg_alpha and reg_lambda
    alpha_grid = gen_grid('reg_alpha', xgb_params, search_spaces, 'log')
    lambda_grid = gen_grid('reg_lambda', xgb_params, search_spaces, 'log')
    reg_alpha, reg_lambda = tune_parameter_pair([('reg_alpha', alpha_grid), ('reg_lambda', lambda_grid)], xgb_params, X, y)
    xgb_params['reg_alpha'] = reg_alpha
    xgb_params['reg_lambda'] = reg_lambda

    # Step 6: Lower training rate
    xgb_params['learning_rate'] *= 0.95

    evaluation_summary = evaluate(X, y, xgb_params, debug=False)
    return xgb_params, evaluation_summary['cv']


if __name__ == '__main__':
    min_year = 2018
    max_year = 2021

    debug = True

    name = f'simple_{min_year}_{max_year}'

    ######################
    # BUILD DATA FROM DB #
    ######################

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

    ####################
    # PARAMETER TUNING #
    ####################

    # Initial parameters
    xgb_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'gamma': 0,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'early_stopping_rounds': 50
    }

    # Search spaces
    search_spaces = {
        'n_estimators': (1, 5000),
        'max_depth': (1, 20),
        'learning_rate': (0, 1),
        'gamma': (0, 20),
        'min_child_weight': (1, 20),
        'subsample': (0, 1),
        'colsample_bytree': (0, 1),
        'reg_alpha': (0, 10),
        'reg_lambda': (1, 20)
    }

    print(xgb_params)
    for i in range(50):
        xgb_params, cv_score = tune_all_parameters(xgb_params, search_spaces, X, y)
        print(xgb_params, cv_score)




