import pandas as pd

import tsfresh as tsf
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

from funding.io import load_features, save_features

from interfaces.db import DB
from utils.text.io import log


def build_time_series(min_date, max_date, concept_ids, debug=False):
    """
    Builds time series DataFrame with data from database, for the specified dates and concepts.

    Args:
        min_date (string): Only data with date >= min_date is considered. Format yyyy-mm-dd.
        max_date (string): Only data with date < max_date is considered. Format yyyy-mm-dd.
        concept_ids (list[int]): Only consider data related with concepts whose id is in this list.
        debug (boolean): Whether data is printed at each step. Default: False.

    Returns (pd.DataFrame): A DataFrame with columns `concept_id`, `concept_name`, `time` and `amount`. Each row
        represents the aggregated investment in USD (`amount`) for a given concept (`concept_id`, `concept_name`)
        during a time period (`time`).
    """

    pd.set_option('display.width', 320)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 10)

    db = DB()

    # Check years
    assert min_date <= max_date, 'max_date must be greater than or equal to min_date'

    ##################
    # FUNDING ROUNDS #
    ##################

    # Get funding rounds in time window
    frs = pd.DataFrame(db.get_funding_rounds(min_date, max_date), columns=['fr_id', 'time', 'amount'])
    log(frs, debug)

    # Extract list of funding round ids
    fr_ids = list(set(frs['fr_id']))
    log(f'Got {len(fr_ids)} funding rounds!', debug)

    #############
    # INVESTEES #
    #############

    # Get association investees <-> funding rounds
    investees_frs = pd.DataFrame(db.get_investees_funding_rounds(fr_ids=fr_ids), columns=['investee_id', 'fr_id'])
    log(investees_frs, debug)

    # Extract list of investee ids
    investee_ids = list(set(investees_frs['investee_id']))
    log(f'Got {len(investee_ids)} investees!', debug)

    # Get investees
    investees = pd.DataFrame(db.get_organisations(org_ids=investee_ids), columns=['investee_id', 'investee_name'])
    log(investees, debug)

    ############
    # CONCEPTS #
    ############

    # Get association concepts <-> investees
    concepts_investees = pd.DataFrame(db.get_concepts_organisations(concept_ids=concept_ids, org_ids=investee_ids), columns=['concept_id', 'investee_id'])
    log(concepts_investees, debug)

    # Get concepts
    concepts = pd.DataFrame(db.get_concepts(concept_ids), columns=['concept_id', 'concept_name'])
    log(concepts, debug)

    ###############
    # TIME SERIES #
    ###############

    # Merge funding rounds with investees
    time_series = pd.merge(frs, investees_frs, how='inner', on='fr_id')
    time_series = pd.merge(time_series, investees, how='left', on='investee_id')
    log(time_series, debug)

    # Merge with concepts
    time_series = pd.merge(time_series, concepts_investees, how='inner', on='investee_id')
    time_series = pd.merge(time_series, concepts, how='left', on='concept_id')
    log(time_series, debug)

    # Aggregate by concept and time
    time_series = time_series[['concept_id', 'time', 'amount']]
    time_series = time_series.groupby(by=['concept_id', 'time'], as_index=False).sum()
    log(time_series, debug)

    # Complete missing data (e.g. time periods with no data for a concept)
    time_periods = time_period_range(min(time_series['time']), max(time_series['time']))
    skeleton = pd.merge(concepts, pd.DataFrame({'time': time_periods}), how='cross')
    time_series = pd.merge(skeleton, time_series, how='left', on=['concept_id', 'time'])
    log(time_series, debug)

    # Fill NA values
    time_series = time_series.fillna(0)
    log(time_series, debug)

    return time_series


def next_time_period(time_period):
    # Split year and quarter and convert to int
    year, quarter = time_period.split('-Q')
    year = int(year)
    quarter = int(quarter)

    # Update year and quarter with modular arithmetic, conjugating with -1 to move between [0, 3] to [1, 4].
    next_quarter = (quarter - 1) + 1 % 4 + 1
    next_year = year + quarter // 4

    return f'{next_year}-Q{next_quarter}'


def prev_time_period(time_period):
    # Split year and quarter and convert to int
    year, quarter = time_period.split('-Q')
    year = int(year)
    quarter = int(quarter)

    # Update year and quarter with modular arithmetic, conjugating with -1 to move between [0, 3] to [1, 4].
    prev_quarter = (quarter - 1) - 1 % 4 + 1
    prev_year = year - prev_quarter // 4

    return f'{prev_year}-Q{prev_quarter}'


def time_period_range(min_time_period, max_time_period):
    assert min_time_period <= max_time_period, 'min_time_period must be less than or equal to max_time_period'

    if min_time_period == max_time_period:
        return [min_time_period]

    # Split years and quarters and convert to int
    min_year, min_quarter = min_time_period.split('-Q')
    min_year = int(min_year)

    max_year, max_quarter = max_time_period.split('-Q')
    max_year = int(max_year)

    # Iterate through possible time periods and add them if they are between min and max
    time_periods = []
    for year in range(min_year, max_year + 1):
        for quarter in [1, 2, 3, 4]:
            time_period = f'{year}-Q{quarter}'
            if min_time_period <= time_period <= max_time_period:
                time_periods.append(time_period)

    return time_periods


def split_last_time_period(df):
    # Extract response variable
    last_time_period = max(df['time'])
    y = df[df['time'] == last_time_period]
    df = df[df['time'] < last_time_period]

    # Set concept_id as index
    y.index = y['concept_id']

    # Convert y to pd.Series
    y = y['amount']

    return df, y


def combine_last_time_period(df, y):
    # Convert y to pd.DataFrame
    y = pd.DataFrame(y, columns=['amount'])

    # Add index as column
    y['concept_id'] = y.index

    # Add time column
    y['time'] = next_time_period(max(df['time']))

    # Add concept names
    concepts = df[['concept_id', 'concept_name']].drop_duplicates()
    y = pd.merge(y, concepts, how='left', on='concept_id')

    # Fix column order
    y = y[df.columns]

    # Combine DataFrames, sort values and reset index
    combined_df = pd.concat([df, y])
    combined_df = combined_df.sort_values(by=['concept_id', 'time']).reset_index(drop=True)

    return combined_df


def build_data(min_date=None, max_date=None, concept_ids=None, features_name=None, split_y=False, debug=False):
    # Normalize input:
    # * If features_name is not provided, make sure the years are, and load the concept_ids if needed.
    # * Else, use years if provided or get years and concepts from the attributes otherwise.
    if features_name is None:
        assert min_date is not None and max_date is not None, f'If features_name is not provided, min_date and max_date must be provided'
    else:
        features, attributes = load_features(features_name)

        if min_date is None or max_date is None:
            log(f'features_name are provided but some of min_date, max_date are not. Using years and concepts from attributes in {features_name}...', debug)
            min_date = attributes['min_date']
            max_date = attributes['max_date']
            concept_ids = attributes['concept_ids']
        else:
            log(f'features_name are provided, min_date, max_date are provided too. Using min_date={min_date} and max_date={max_date} and ignoring years and concepts from attributes in {features_name}...', debug)

    if concept_ids is None:
        db = DB()
        concept_ids = db.get_crunchbase_concept_ids()

    # Create time series with data from database
    log(f'Creating time series for time window [{min_date}, {max_date}) and {len(concept_ids)} concepts...', debug)
    df = build_time_series(min_date, max_date, concept_ids=concept_ids, debug=False)
    print(f'Created df with shape {df.shape}')

    # Split df rows by extracting last time period
    if split_y:
        log(f'Splitting y from last time period data', debug)
        df, y = split_last_time_period(df)

    # Extract features
    if features_name is None:
        log(f'Extracting all features...', debug)
        X = extract_features(df)
    else:
        log(f'Extracting features {features_name}...', debug)
        X = extract_features(df, features)

    if split_y:
        return X, y
    else:
        return X


def extract_features(df, features=None):
    if features is None:
        features = {'amount': EfficientFCParameters()}

    print(f'Ramtin shape of df: {df.shape}')
    X = tsf.extract_features(df, column_id='concept_id', column_sort='time', column_value='amount',
                         default_fc_parameters={},
                         kind_to_fc_parameters=features,
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


def get_feature_importances(X, model):
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    return feature_importances


def create_feature_set(min_date, max_date, concept_ids=None, debug=False):
    # Sanity check
    assert min_date < max_date, f'min_year ({min_date}) should be lower than max_year ({max_date})'

    # Build data
    log(f'Building data for time window [{min_date}, {max_date}) and all features...', debug)
    X, y = build_data(min_date=min_date, max_date=max_date, concept_ids=concept_ids, split_y=True, debug=True)

    # Select most relevant features
    log(f'Extracting features and selecting the most relevant ones...', debug)
    X = select_features(X, y)

    # Save features
    if concept_ids is None:
        name = f'{min_date}-{max_date}-all'
    else:
        name = f'{min_date}-{max_date}-{sum(concept_ids)}'
    log(f'Saving features to disk under the name "{name}"...', debug)
    features = from_columns(X)
    attributes = {
        'min_date': min_date,
        'max_date': max_date,
        'concept_ids': concept_ids
    }
    save_features(features, attributes, name=name)

    return name


