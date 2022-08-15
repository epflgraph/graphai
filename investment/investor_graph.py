import pandas as pd

from itertools import combinations

from interfaces.db import DB

import networkx as nx


class InvestorGraph:

    def __init__(self, min_date, max_date):
        self.min_date = min_date
        self.max_date = max_date

        # Init dummy values
        self.frs = pd.DataFrame()
        self.fr_ids = []

        self.investors_frs = pd.DataFrame()

        self.investor_pairs = {}

        self.investor_edges = []

        self.G = nx.Graph()
        self.n_nodes = 0
        self.n_edges = 0
        self.cc_sizes = []

        # Populate variables with actual data from database
        print('Retrieving funding rounds...')
        self.retrieve_funding_rounds()
        print('Retrieving investors...')
        self.retrieve_investors()
        print('Computing investor pairs...')
        self.compute_investor_pairs()
        print('Computing investor edges...')
        self.compute_investor_edges()
        print('Creating investor graph...')
        self.create_investor_graph()

    def retrieve_funding_rounds(self):
        """
        Sets variables self.frs and self.fr_ids.
        """

        # Instantiate db interface to communicate with database
        db = DB()

        # Retrieve funding rounds in time window
        fields = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'CB_InvestorCount']
        columns = ['fr_id', 'fr_date', 'fr_amount', 'fr_n_investors']
        self.frs = pd.DataFrame(db.get_funding_rounds(min_date=self.min_date, max_date=self.max_date, fields=fields), columns=columns)

        # Create derived variable
        self.frs['fr_amount_per_investor'] = self.frs['fr_amount'] / self.frs['fr_n_investors']

        # Extract list of funding round ids in time window
        self.fr_ids = list(self.frs['fr_id'])

    def retrieve_investors(self):
        """
        Sets variable self.investors_frs. Assumes self.fr_ids is set.
        """

        # Instantiate db interface to communicate with database
        db = DB()

        # Retrieve organization and person investors and combine them in a single DataFrame
        org_investors_frs = pd.DataFrame(db.get_org_investors_funding_rounds(fr_ids=self.fr_ids), columns=['investor_id', 'fr_id'])
        org_investors_frs['investor_type'] = 'Organization'
        person_investors_frs = pd.DataFrame(db.get_person_investors_funding_rounds(fr_ids=self.fr_ids), columns=['investor_id', 'fr_id'])
        person_investors_frs['investor_type'] = 'Person'
        self.investors_frs = pd.concat([org_investors_frs, person_investors_frs])

    def compute_investor_pairs(self):
        """
        Sets variable self.investor_pairs. Assumes self.frs and self.investors_frs are set.
        """

        # Create DataFrame with all investor relationships.
        # Two investors are related if they have participated in at least one funding round together
        # in the given time period.
        def combine(group):
            return pd.DataFrame.from_records(combinations(group['investor_id'], 2))

        investor_pairs = self.investors_frs.groupby('fr_id').apply(combine).reset_index(level=0)
        investor_pairs.columns = ['fr_id', 'investor_id_1', 'investor_id_2']

        # Add funding round information
        investor_pairs = investor_pairs.merge(self.frs, how='left', on='fr_id')

        # Convert to list of dictionaries to create graph
        self.investor_pairs = investor_pairs.to_dict(orient='records')

    def compute_investor_edges(self):
        """
        Sets variable self.investor_edges. Assumes self.investor_pairs is set.
        """

        # Group by investor pair and include aggregated funding rounds as history data of the investor pair
        investor_edges = {}
        for pair in self.investor_pairs:
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
        self.investor_edges = [(investor_edges[key]['s'], investor_edges[key]['t'], {'hist': investor_edges[key]['d']}) for key in investor_edges]

    def create_investor_graph(self):
        """
        Sets variables self.G, self.n_nodes, self.n_edges and self.cc_sizes. Assumes self.investor_edges is set.
        """

        # Create graph and populate it with data from investor edges
        self.G = nx.Graph()
        self.G.add_edges_from(self.investor_edges)
        self.G.remove_nodes_from(list(nx.isolates(self.G)))

        self.n_nodes = len(self.G.nodes)
        self.n_edges = len(self.G.edges)

        self.cc_sizes = [len(c) for c in sorted(nx.connected_components(self.G), key=len, reverse=True)]

    def __repr__(self):
        return f'InvestorGraph for time range [{self.min_date}, {self.max_date}) [n_nodes: {self.n_nodes}, n_edges: {self.n_edges}, n_ccs: {len(self.cc_sizes)}, cc_sizes: {self.cc_sizes[:5] + [...]}]'
