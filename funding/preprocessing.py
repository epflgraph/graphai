import pandas as pd

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
