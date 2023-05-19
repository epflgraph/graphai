import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate

from xgboost import XGBRegressor

from funding.data_processing import build_data
from funding.io import save_xgb_params

from utils.text.io import log


def evaluate(X, y, xgb_params=None, debug=False):
    # Split train/test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    if xgb_params is None:
        xgb_params = {}

    # Create model
    model = XGBRegressor(**xgb_params)

    # Create cross-validation object
    n_splits = 10
    n_repeats = 3
    # cv = RepeatedKFold(n_splits=5, n_repeats=6, random_state=0)
    cv = RepeatedKFold(n_splits=5, n_repeats=6)

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


def restrict_search_space(param, value, search_spaces, type):
    param_min = search_spaces[param][0]
    param_max = search_spaces[param][1]

    if is_stable(param, search_spaces):
        return search_spaces

    diff = abs(param_max - param_min)
    should_be_stable = (type == 'int' and diff < 1) or (type == 'float' and diff < 0.05) or (type == 'log' and diff < 0.05) or (type == '1+log' and diff < 0.05)

    if should_be_stable:
        search_spaces[param] = (param_min, param_max, 'stable')
        return search_spaces

    grid = gen_grid(param, search_spaces, type)

    if len(grid) <= 1:
        return search_spaces

    lower_grid = [x for x in grid if x <= value]
    upper_grid = [x for x in grid if x >= value]

    lower_grid = sorted(list(set(lower_grid + [value])))
    upper_grid = sorted(list(set([value] + upper_grid)))

    n_lower = len(lower_grid)
    n_upper = len(upper_grid)

    if n_lower >= 2 and n_lower >= n_upper:
        param_min = lower_grid[1]

    if n_upper >= 2 and n_lower <= n_upper:
        param_max = upper_grid[-2]

    search_spaces[param] = (param_min, param_max, 'open')

    return search_spaces


def gen_grid(param, search_spaces, type, n_steps=7):
    param_min = search_spaces[param][0]
    param_max = search_spaces[param][1]
    param_status = search_spaces[param][2]

    if param_status == 'stable':
        return list(set([param_min, param_max]))

    if type == 'int':
        grid = [int(np.round(x)) for x in np.linspace(param_min, param_max, n_steps)]
    elif type == 'float':
        grid = [x for x in np.linspace(param_min, param_max, n_steps)]
    elif type == 'log':
        if param_min == 0:
            epsilon = 1e-6
            grid = [0] + list(np.geomspace(epsilon, param_max, n_steps))
        else:
            grid = list(np.geomspace(param_min, param_max, n_steps))
    elif type == '1+log':
        if param_min == 1:
            epsilon = 1e-6
            grid = [1] + [1 + x for x in np.geomspace(epsilon, param_max - 1, n_steps)]
        else:
            grid = [1 + x for x in np.geomspace(param_min - 1, param_max - 1, n_steps)]

    else:
        grid = []

    grid = list(set(grid))
    grid = [x for x in grid if param_min <= x <= param_max]

    return grid


def is_stable(param, search_spaces):
    return search_spaces[param][2] == 'stable'


def tune_all_parameters(xgb_params, search_spaces, X, y):
    # Step 1: Tune max_depth and min_child_weight
    while not is_stable('max_depth', search_spaces) or not is_stable('min_child_weight', search_spaces):
        # Create search grids based on search spaces
        md_grid = gen_grid('max_depth', search_spaces, 'int')
        mcw_grid = gen_grid('min_child_weight', search_spaces, 'int')

        # Perform parameter tuning with given grids
        max_depth, min_child_weight = tune_parameter_pair([('max_depth', md_grid), ('min_child_weight', mcw_grid)], xgb_params, X, y)

        # Update optimal parameters
        xgb_params['max_depth'] = max_depth
        xgb_params['min_child_weight'] = min_child_weight

        # Update search spaces based on optimal parameters
        search_spaces = restrict_search_space('max_depth', max_depth, search_spaces, 'int')
        search_spaces = restrict_search_space('min_child_weight', min_child_weight, search_spaces, 'int')

        # Compute the cv score and print report
        cv_score = evaluate(X, y, xgb_params, debug=False)['cv']
        report(xgb_params, search_spaces, cv_score)

    # Step 2: Tune gamma
    while not is_stable('gamma', search_spaces):
        # Create search grid based on search space
        gamma_grid = gen_grid('gamma', search_spaces, 'log')

        # Perform parameter tuning with given grid
        gamma = tune_parameter_single([('gamma', gamma_grid)], xgb_params, X, y)

        # Update optimal parameter
        xgb_params['gamma'] = gamma

        # Update search space based on optimal parameter
        search_spaces = restrict_search_space('gamma', gamma, search_spaces, 'log')

        # Compute the cv score and print report
        cv_score = evaluate(X, y, xgb_params, debug=False)['cv']
        report(xgb_params, search_spaces, cv_score)

    # Step 3: Tune subsample and colsample_bytree
    while not is_stable('subsample', search_spaces) or not is_stable('colsample_bytree', search_spaces):
        # Create search grids based on search spaces
        ss_grid = gen_grid('subsample', search_spaces, 'float')
        cs_grid = gen_grid('colsample_bytree', search_spaces, 'float')

        # Perform parameter tuning with given grids
        subsample, colsample_bytree = tune_parameter_pair([('subsample', ss_grid), ('colsample_bytree', cs_grid)], xgb_params, X, y)

        # Update optimal parameters
        xgb_params['subsample'] = subsample
        xgb_params['colsample_bytree'] = colsample_bytree

        # Update search spaces based on optimal parameters
        search_spaces = restrict_search_space('subsample', subsample, search_spaces, 'float')
        search_spaces = restrict_search_space('colsample_bytree', colsample_bytree, search_spaces, 'float')

        # Compute the cv score and print report
        cv_score = evaluate(X, y, xgb_params, debug=False)['cv']
        report(xgb_params, search_spaces, cv_score)

    # Step 4: Tune reg_alpha and reg_lambda
    while not is_stable('reg_alpha', search_spaces) or not is_stable('reg_lambda', search_spaces):
        # Create search grids based on search spaces
        alpha_grid = gen_grid('reg_alpha', search_spaces, 'log')
        lambda_grid = gen_grid('reg_lambda', search_spaces, '1+log')

        # Perform parameter tuning with given grids
        reg_alpha, reg_lambda = tune_parameter_pair([('reg_alpha', alpha_grid), ('reg_lambda', lambda_grid)], xgb_params, X, y)

        # Update optimal parameters
        xgb_params['reg_alpha'] = reg_alpha
        xgb_params['reg_lambda'] = reg_lambda

        # Update search spaces based on optimal parameters
        search_spaces = restrict_search_space('reg_alpha', reg_alpha, search_spaces, 'log')
        search_spaces = restrict_search_space('reg_lambda', reg_lambda, search_spaces, '1+log')

        # Compute the cv score and print report
        cv_score = evaluate(X, y, xgb_params, debug=False)['cv']
        report(xgb_params, search_spaces, cv_score)

    return xgb_params, search_spaces


def update_n_estimators(X, y, xgb_params, initial_n_estimators=1000):
    xgb_params['n_estimators'] = initial_n_estimators
    avg_n_trees = evaluate(X, y, xgb_params, debug=False)['avg_n_trees']
    xgb_params['n_estimators'] = avg_n_trees

    return xgb_params


def report(xgb_params, search_spaces, cv_score):
    print()
    for param in search_spaces:
        print(param, xgb_params[param], search_spaces[param])
    print(cv_score)


def create_tuned_xgb_params(features_name, debug=False):
    # Build data
    X, y = build_data(features_name=features_name, split_y=True, debug=debug)

    ####################
    # PARAMETER TUNING #
    ####################

    # Initial parameters, with learning_rate = 0.1 and n_estimators = 1000
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

    # Get optimal number of estimators for the given parameters and update parameter
    xgb_params = update_n_estimators(X, y, xgb_params)

    # Define search spaces for each parameter and set all their status to 'open'
    search_spaces = {
        'n_estimators': (1, 5000, 'open'),
        'max_depth': (1, 20, 'open'),
        'learning_rate': (0, 1, 'open'),
        'gamma': (0, 10, 'open'),
        'min_child_weight': (1, 20, 'open'),
        'subsample': (0.5, 1, 'open'),
        'colsample_bytree': (0.5, 1, 'open'),
        'reg_alpha': (0, 10, 'open'),
        'reg_lambda': (1, 10, 'open')
    }

    # Launch tuning of all parameters
    xgb_params, search_spaces = tune_all_parameters(xgb_params, search_spaces, X, y)

    # Lower learning rate and get new optimal number of estimators
    xgb_params['learning_rate'] = 0.01
    xgb_params = update_n_estimators(X, y, xgb_params)

    # Disable early stopping
    del xgb_params['early_stopping_rounds']

    # Print final report
    evaluation_summary = evaluate(X, y, xgb_params, debug=False)
    report(xgb_params, search_spaces, evaluation_summary['cv'])

    # Create model with the tuned parameters
    xgb_params_name = 'tuned'
    save_xgb_params(xgb_params, evaluation_summary, features_name, xgb_params_name)

    return xgb_params_name
