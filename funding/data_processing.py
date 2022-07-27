import pandas as pd

import tsfresh as tsf
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

from funding.io import load_features, save_features

from interfaces.db import DB
from utils.text.io import log


def build_time_series(min_year, max_year, concept_ids, debug=False):
    pd.set_option('display.width', 320)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 10)

    db = DB()

    # Check years
    assert min_year <= max_year, 'max_year must be greater than or equal to min_year'

    ##################
    # FUNDING ROUNDS #
    ##################

    # Get funding rounds in time window
    frs = pd.DataFrame(db.get_funding_rounds(min_year, max_year), columns=['fr_id', 'year', 'amount'])
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

    # Aggregate by concept and year
    time_series = time_series[['concept_id', 'year', 'amount']]
    time_series = time_series.groupby(by=['concept_id', 'year'], as_index=False).sum()
    log(time_series, debug)

    # Complete missing data (e.g. years with no data for a concept)
    skeleton = pd.merge(concepts, pd.DataFrame({'year': range(min_year, max_year + 1)}), how='cross')
    time_series = pd.merge(skeleton, time_series, how='left', on=['concept_id', 'year'])
    log(time_series, debug)

    # Fill NA values
    time_series = time_series.fillna(0)
    log(time_series, debug)

    return time_series


def split_last_year(df, last_year):
    # Extract response variable
    y = df[df['year'] == last_year]
    df = df[df['year'] < last_year]

    # Set concept_id as index
    y.index = y['concept_id']

    # Convert y to pd.Series
    y = y['amount']

    return df, y


def combine_last_year(df, y, last_year):
    # Convert y to pd.DataFrame
    y = pd.DataFrame(y, columns=['amount'])

    # Add index as column
    y['concept_id'] = y.index

    # Add year column
    y['year'] = last_year

    # Add concept names
    concepts = df[['concept_id', 'concept_name']].drop_duplicates()
    y = pd.merge(y, concepts, how='left', on='concept_id')

    # Fix column order
    y = y[df.columns]

    # Combine DataFrames, sort values and reset index
    combined_df = pd.concat([df, y])
    combined_df = combined_df.sort_values(by=['concept_id', 'year']).reset_index(drop=True)

    return combined_df


def build_data(min_year=None, max_year=None, concept_ids=None, features_name=None, split_y=False, debug=False):
    # Normalize input:
    # * If features_name is not provided, make sure the years are, and load the concept_ids if needed.
    # * Else, use years if provided or get years and concepts from the attributes otherwise.
    if features_name is None:
        assert min_year is not None and max_year is not None, f'If features_name is not provided, min_year and max_year must be provided'

        if concept_ids is None:
            db = DB()
            concept_ids = db.get_crunchbase_concept_ids()
    else:
        features, attributes = load_features(features_name)

        if min_year is None or max_year is None:
            log(f'features_name are provided but some of min_year, max_year are not. Using years and concepts from attributes in {features_name}...', debug)
            min_year = attributes['min_year']
            max_year = attributes['max_year']
            concept_ids = attributes['concept_ids']
        else:
            log(f'features_name are provided, min_year, max_year are provided too. Using min_year={min_year} and max_year={max_year} and ignoring years and concepts from attributes in {features_name}...', debug)

            if concept_ids is None:
                db = DB()
                concept_ids = db.get_crunchbase_concept_ids()

    # Create time series with data from database
    log(f'Creating time series for time window {min_year}-{max_year}...', debug)
    df = build_time_series(min_year, max_year, concept_ids=concept_ids, debug=False)

    # Split df rows into < max_year and = max_year
    if split_y:
        log(f'Splitting y from last year data', debug)
        df, y = split_last_year(df, max_year)

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
        features = {'amount': ComprehensiveFCParameters()}

    X = tsf.extract_features(df, column_id='concept_id', column_sort='year', column_value='amount',
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


def create_feature_set(min_year, max_year, concept_ids=None, debug=False):
    # Sanity check
    assert min_year < max_year, f'min_year ({min_year}) should be lower than max_year ({max_year})'

    # Build data
    log(f'Building data for time window {min_year}-{max_year} and all features...', debug)
    X, y = build_data(min_year=min_year, max_year=max_year, concept_ids=concept_ids, split_y=True, debug=False)

    # Select most relevant features
    log(f'Extracting features and selecting the most relevant ones...', debug)
    X = select_features(X, y)

    # Save features
    if concept_ids is None:
        name = f'{min_year}-{max_year}-all'
    else:
        name = f'{min_year}-{max_year}-{sum(concept_ids)}'
    log(f'Saving features to disk under the name "{name}"...', debug)
    features = from_columns(X)
    attributes = {
        'min_year': min_year,
        'max_year': max_year,
        'concept_ids': concept_ids
    }
    save_features(features, attributes, name=name)

    return name


