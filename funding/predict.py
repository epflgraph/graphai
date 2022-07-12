import pandas as pd
import matplotlib.pyplot as plt

from interfaces.db import DB

top_concept_ids = [5309, 36674345, 282635, 39388, 261925, 433425, 9611, 2861, 14539, 59252]
bottom_concept_ids = [4015968, 25405, 32558, 8965447, 21197, 30589, 1030104, 26201, 42125403, 31775]

# Artificial intelligence, Fashion, Genetics, Graphic design, Hedge fund, Business incubator, Journalism,
# Social network, Educational technology, Cloud computing, Public transport, Big data, Bitcoin, Biometrics,
# Clean technology, Crowdfunding, ESports, Facial recognition system, Coffee, LGBT, QR code, Digital marketing
cherrypicked_concept_ids = [1164, 11657, 12266, 12799, 14412, 1526031, 15928,
                            34327569, 1944675, 19541494, 26162030, 27051151, 28249265, 290622,
                            4835266, 48505834, 564204, 602401, 604727, 66936, 828436, 9933471]

# Artificial intelligence, Fashion, Genetics, Educational technology, Cloud computing,
# Public transport, Bitcoin, Clean technology, ESports, Coffee
reduced_cherrypicked_concept_ids = [1164, 11657, 12266, 1944675, 19541494,
                                    26162030, 28249265, 4835266, 564204, 604727]

# concept_ids = bottom_concept_ids[-5:]
# concept_ids = top_concept_ids[:5]
concept_ids = reduced_cherrypicked_concept_ids


def log(msg, debug):
    if debug:
        print(msg)


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
    time_series = time_series[['concept_id', 'concept_name', 'year', 'amount']]
    time_series = time_series.groupby(by=['concept_id', 'concept_name', 'year'], as_index=False).sum()
    log(time_series, debug)

    # Fill NA values
    time_series = time_series.fillna(0)
    log(time_series, debug)

    return time_series


build_time_series(2019, 2021, concept_ids, debug=True)


