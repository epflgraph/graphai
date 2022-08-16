import pandas as pd

import networkx as nx

from itertools import combinations

from interfaces.db import DB

from utils.time.stopwatch import Stopwatch


def retrieve_funding_rounds(min_date, max_date):

    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve funding rounds in time window
    fields = ['FundingRoundID']
    columns = ['fr_id']
    frs = pd.DataFrame(db.get_funding_rounds(min_date=min_date, max_date=max_date, fields=fields), columns=columns)

    return frs


def retrieve_investors(frs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Extract list of funding round ids
    fr_ids = list(frs['fr_id'])

    # Retrieve organization and person investors and combine them in a single DataFrame
    org_investors_frs = pd.DataFrame(db.get_org_investors_funding_rounds(fr_ids=fr_ids), columns=['investor_id', 'fr_id'])
    org_investors_frs['investor_type'] = 'Organization'
    person_investors_frs = pd.DataFrame(db.get_person_investors_funding_rounds(fr_ids=fr_ids), columns=['investor_id', 'fr_id'])
    person_investors_frs['investor_type'] = 'Person'
    investors_frs = pd.concat([org_investors_frs, person_investors_frs])

    return investors_frs


def insert_investors(investors_frs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Drop duplicate investors
    investors = investors_frs[['investor_id', 'investor_type']].drop_duplicates()

    # Insert into DB
    db.create_table_Nodes_N_Investor(investors)


def compute_investor_pairs(investors_frs):
    # Create DataFrame with all investor relationships.
    # Two investors are related if they have participated in at least one funding round together
    # in the given time period.
    def combine(group):
        return pd.DataFrame.from_records(combinations(group['investor_id'], 2))

    investor_pairs = investors_frs.groupby('fr_id').apply(combine).reset_index(level=0)
    investor_pairs.columns = ['fr_id', 'source_investor_id', 'target_investor_id']

    # Duplicate DataFrame flipping source and target, since the relation is symmetrical
    first_half = pd.DataFrame(investor_pairs[['fr_id', 'source_investor_id', 'target_investor_id']].values, columns=['fr_id', 'source_investor_id', 'target_investor_id'])
    second_half = pd.DataFrame(investor_pairs[['fr_id', 'target_investor_id', 'source_investor_id']].values, columns=['fr_id', 'source_investor_id', 'target_investor_id'])
    return pd.concat([first_half, second_half]).reset_index(drop=True)


def insert_investor_edges(investor_pairs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Drop duplicate investors
    investors = investor_pairs[['source_investor_id', 'target_investor_id']].drop_duplicates()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Investor(investors)


def retrieve_concepts(frs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Extract list of funding round ids
    fr_ids = list(frs['fr_id'])

    # Retrieve investees
    investees_frs = pd.DataFrame(db.get_investees_funding_rounds(fr_ids=fr_ids), columns=['investee_id', 'fr_id'])

    # Extract list of investee ids
    investee_ids = list(investees_frs['investee_id'])

    # Retrieve investees' concepts
    concepts_investees = pd.DataFrame(db.get_concepts_organisations(org_ids=investee_ids), columns=['concept_id', 'investee_id'])

    # Join frs and concepts through investees
    concepts_frs = pd.merge(investees_frs, concepts_investees, how='inner', on='investee_id')
    concepts_frs = concepts_frs[['fr_id', 'concept_id']]

    return concepts_frs


def compute_investor_concept_pairs(investors_frs, concepts_frs):
    investors_concepts = pd.merge(investors_frs, concepts_frs, how='inner', on='fr_id')
    investors_concepts = investors_concepts[['investor_id', 'investor_type', 'concept_id']].drop_duplicates()

    return investors_concepts


def insert_investor_concept_edges(investors_concepts):
    # Instantiate db interface to communicate with database
    db = DB()

    investors_concepts_edges = investors_concepts[['investor_id', 'concept_id']]

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Concept(investors_concepts_edges)











def compute_investor_edges(investor_pairs):
    # Convert to list of dictionaries to create graph
    investor_pairs = investor_pairs.to_dict(orient='records')

    # Group by investor pair and include aggregated funding rounds as history data of the investor pair
    investor_edges = {}
    for pair in investor_pairs:
        key = f"{pair['investor_id_1']}x{pair['investor_id_2']}"

        fr = {
            'fr_id': pair['fr_id'],
            'date': pair['fr_date'],
            'amount': pair['fr_amount'],
            'n_investors': pair['fr_n_investors'],
            'amount_per_investor': pair['fr_amount_per_investor']
        }

        if key in investor_edges:
            investor_edges[key]['d'].append(fr)
        else:
            investor_edges[key] = {
                's': pair['investor_id_1'],
                't': pair['investor_id_2'],
                'd': [fr]
            }

    # Reshape as networkx container of edges
    investor_edges = [(investor_edges[key]['s'], investor_edges[key]['t'], {'hist': investor_edges[key]['d']}) for key in investor_edges]

    return investor_edges


def create_investor_graph(investor_edges):
    # Create graph and populate it with data from investor edges
    G = nx.Graph()
    G.add_edges_from(investor_edges)
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def main():
    sw = Stopwatch()

    min_date = '2021-11-01'
    max_date = '2022-01-01'

    print('0. Retrieving funding rounds...')
    frs = retrieve_funding_rounds(min_date, max_date)
    sw.tick()

    print('1. Retrieving investors...')
    investors_frs = retrieve_investors(frs)
    sw.tick()

    print('2. Inserting investors into database...')
    insert_investors(investors_frs)
    sw.tick()

    print('3. Computing investor pairs...')
    investor_pairs = compute_investor_pairs(investors_frs)
    sw.tick()

    print('4. Inserting investor edges into database...')
    insert_investor_edges(investor_pairs)
    sw.tick()

    print('5. Retrieving concepts...')
    concepts_frs = retrieve_concepts(frs)
    sw.tick()

    print('6. Computing investor-concept pairs...')
    investors_concepts = compute_investor_concept_pairs(investors_frs, concepts_frs)
    sw.tick()

    print('7. Inserting investor-concept edges into database...')
    insert_investor_concept_edges(investors_concepts)

    # print('Computing investor edges...')
    # investor_edges = compute_investor_edges(investor_pairs)
    # print('Creating investor graph...')
    # G = create_investor_graph(investor_edges)
    #
    # n_nodes = len(G.nodes)
    # n_edges = len(G.edges)
    # cc_sizes = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    #
    # print(f'Investor graph for time range [{min_date}, {max_date}) [n_nodes: {n_nodes}, n_edges: {n_edges}, n_ccs: {len(cc_sizes)}, cc_sizes: {cc_sizes[:5] + [...]}]')

    sw.report()

    # nx.draw_spring(G, node_size=1)
    # import matplotlib.pyplot as plt
    # plt.show()


if __name__ == '__main__':
    main()
