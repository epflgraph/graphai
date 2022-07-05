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


def build_dataframe(concept_ids, debug=False):
    pd.set_option('display.width', 320)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 10)

    db = DB()

    # Get concepts with id in starting list
    concepts = pd.DataFrame(db.get_concepts(concept_ids), columns=['concept_id', 'concept_name'])
    log(concepts, debug)

    # Extract list of concept ids
    concept_ids = list(set(concepts['concept_id']))
    log(f'Got {len(concept_ids)} concepts!', debug)

    # Add association concepts <-> organisations
    concepts_orgs = pd.DataFrame(db.get_concept_organisations(concept_ids), columns=['concept_id', 'org_id'])
    concepts_orgs = pd.merge(concepts, concepts_orgs, how='inner', on='concept_id')
    log(concepts_orgs, debug)

    # Extract list of organisation ids
    org_ids = list(set(concepts_orgs['org_id']))
    log(f'Got {len(org_ids)} organisations!', debug)

    # Add association funded organisations <-> funding rounds
    funded_orgs_frs = pd.DataFrame(db.get_beneficiary_funding_rounds(org_ids), columns=['org_id', 'fr_id'])
    concepts_orgs_frs = pd.merge(concepts_orgs, funded_orgs_frs, how='inner', on='org_id')
    log(concepts_orgs_frs, debug)

    # Extract list of funding round ids
    fr_ids = list(set(concepts_orgs_frs['fr_id']))
    log(f'Got {len(fr_ids)} funding rounds!', debug)

    # Add funding round dates and amounts
    frs = pd.DataFrame(db.get_funding_rounds(fr_ids), columns=['fr_id', 'date', 'amount'])
    concepts_orgs_frs = pd.merge(concepts_orgs_frs, frs, how='inner', on='fr_id')
    log(concepts_orgs_frs, debug)

    # Derive year and month from date
    concepts_orgs_frs['year'] = concepts_orgs_frs['date'].astype(str).str.split('-').str[0].astype(int)
    concepts_orgs_frs['month'] = concepts_orgs_frs['date'].astype(str).str.split('-').str[1].astype(int)

    # Compute month_id as time variable (1 means January of first year, 13 means January of second year, etc.)
    min_year = max(2000, min(concepts_orgs_frs['year']))
    max_year = min(2021, max(concepts_orgs_frs['year']))
    concepts_orgs_frs['month_id'] = (concepts_orgs_frs['year'] - min_year) * 12 + concepts_orgs_frs['month']
    concepts_orgs_frs = concepts_orgs_frs.drop('date', axis=1)
    log(concepts_orgs_frs, debug)

    # Add investors
    fr_investors = pd.DataFrame(db.get_funding_round_investors(fr_ids), columns=['fr_id', 'investor_id', 'investor_type'])
    concepts_orgs_frs_investors = pd.merge(concepts_orgs_frs, fr_investors, how='inner', on='fr_id')
    log(concepts_orgs_frs_investors, debug)

    # Extract list of investor ids
    investor_ids = list(set(concepts_orgs_frs_investors['investor_id']))
    log(f'Got {len(investor_ids)} investors!', debug)

    # Group by concept, investor and month id
    investors = concepts_orgs_frs_investors[['concept_id', 'investor_id', 'month_id', 'fr_id', 'amount']]
    investors = investors.groupby(by=['concept_id', 'investor_id', 'month_id'], as_index=False).agg(
        n_frs=('fr_id', 'count'), total_amount=('amount', 'sum'))
    log(investors, debug)

    # Create complete grid of concepts and years
    skeleton = investors[['concept_id', 'investor_id']].drop_duplicates()
    skeleton = skeleton.merge(fr_investors[['investor_id', 'investor_type']].drop_duplicates(), how='left', on='investor_id')
    skeleton = skeleton.merge(pd.DataFrame({'month_id': range(1, (max_year - min_year + 1) * 12)}), how='cross')
    investors = pd.merge(skeleton, investors, how='left', on=['concept_id', 'investor_id', 'month_id'])
    log(investors, debug)

    # Fill NA values
    investors = investors.fillna(0)
    log(investors, debug)

    # Add concept names
    investors = pd.merge(investors, concepts, how='left', on='concept_id')

    # Add organisation investor names
    orgs = pd.DataFrame(db.get_organisations(investor_ids), columns=['investor_id', 'org_name'])
    orgs['investor_type'] = 'organisation'
    investors = pd.merge(investors, orgs, how='left', on=['investor_id', 'investor_type'])
    log(investors, debug)

    # Add person investor names
    people = pd.DataFrame(db.get_people(investor_ids), columns=['investor_id', 'person_name'])
    people['investor_type'] = 'person'
    investors = pd.merge(investors, people, how='left', on=['investor_id', 'investor_type'])
    log(investors, debug)

    # Combine org and person names in one column
    investors['investor_name'] = investors['org_name'].fillna(investors['person_name'])
    log(investors, debug)

    # Rearrange columns
    investors = investors[['concept_id', 'concept_name', 'investor_id', 'investor_name', 'investor_type', 'month_id', 'n_frs', 'total_amount']]
    log(investors, debug)

    return investors


# def extract_features():

build_dataframe(concept_ids, debug=True)
