import pandas as pd

from db_cache_manager.db import DB

from graphai.core.common.config import config


class ConceptsGraph:
    def __init__(self):
        # Flag to know whether we've already loaded the data from the database
        self.loaded = False

        # Object of type pd.DataFrame with columns ['PageID', 'PageTitle'] holding the concepts
        self.concepts = None

        # Object of type pd.DataFrame with columns ['SourcePageID', 'TargetPageID', 'NormalisedScore']
        # holding the scored, undirected concept-concept edges
        self.concepts_concepts = None

        ################################################

        # Object of type pd.DataFrame with columns ['SourcePageID', 'TargetPageID'] holding, for each SourcePageID,
        # a set of all its successors (TargetPageIDs).
        self.successors = None

        # Object of type pd.DataFrame with columns ['SourcePageID', 'TargetPageID'] holding, for each TargetPageID,
        # a set of all its predecessors (SourcePageID).
        self.predecessors = None

        ################################################

        # Set containing the PageIDs of all concepts having an outgoing edge
        self.sources = None

        # Set containing the PageIDs of all concepts having an ingoing edge
        self.targets = None

    def fetch_from_db(self):
        if self.loaded:
            print('Graph already loaded')
            print(self.loaded)
            return

        print('Actually loading the graph tables...')

        db = DB(config['database'])

        print('Loading concept nodes table...', end=' ')

        table_name = 'graph.Nodes_N_Concept'
        fields = ['PageID', 'PageTitle']
        self.concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
        # concept_ids = list(self.concepts['PageID'])

        print('Done')

        print('Loading concept edges table...', end=' ')

        table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
        fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
        # conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
        conditions = {}
        self.concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

        print('Done')

        n_concepts = len(pd.concat([self.concepts_concepts['SourcePageID'], self.concepts_concepts['TargetPageID']]).drop_duplicates())
        print(f'Got {len(self.concepts_concepts)} concept to concept edges among {n_concepts} different concepts.')

        print('Concatenating edges with reverse edges for easy access...')

        self.concepts_concepts = pd.concat([
            self.concepts_concepts,
            self.concepts_concepts.rename(columns={'SourcePageID': 'TargetPageID', 'TargetPageID': 'SourcePageID'})
        ]).reset_index(drop=True)

        # print('Storing derived successors, predecessors, sources and targets...')
        #
        # # Store successor and predecessor sets as attributes
        # self.successors = self.concepts_concepts.groupby(by='SourcePageID').aggregate({'TargetPageID': set})
        # self.predecessors = self.concepts_concepts.groupby(by='TargetPageID').aggregate({'SourcePageID': set})
        # self.sources = set(self.successors.index)
        # self.targets = set(self.predecessors.index)

        # Set the flag to avoid future reloads
        print('Setting graph as loaded...')
        self.loaded = True

    def add_graph_score(self, results, smoothing=True):
        """
        Computes GraphScore for the provided intermediate wikify results.

        Args:
            results (pd.DataFrame): A pandas DataFrame including the columns ['Keywords', 'PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore'].
            smoothing (bool): Whether to apply a transformation to the GraphScore that bumps scores to avoid
            a negative exponential shape. Default: True.

        Returns:
            pd.DataFrame: A pandas DataFrame with the original columns plus ['GraphScore'].
        """

        self.fetch_from_db()

        # Extract concept ids from set of results
        concept_ids = list(results['PageID'].drop_duplicates())

        # Keep only edges whose vertices are in the results' concepts
        concepts_concepts = self.concepts_concepts[
            self.concepts_concepts['SourcePageID'].isin(concept_ids)
            & self.concepts_concepts['TargetPageID'].isin(concept_ids)
        ]

        # Fallback: Unlikely case when no pair of concepts in the results share an edge, so induced graph are isolated vertices
        if len(concepts_concepts) == 0:
            results['GraphScore'] = 1
            return results

        # Aggregate by SourcePageID adding up the scores (concepts_concepts contains edges and their reversed edges)
        concepts_concepts = concepts_concepts.groupby(by='SourcePageID').aggregate(GraphScore=('NormalisedScore', 'sum')).reset_index()
        concepts_concepts = concepts_concepts.rename(columns={'SourcePageID': 'PageID'})

        # Normalise so that scores are in [0, 1]
        concepts_concepts['GraphScore'] = concepts_concepts['GraphScore'] / concepts_concepts['GraphScore'].max()

        # Smooth score if needed using the function f(x) = (2 - x) * x, bumping lower values to avoid very low scores
        if smoothing:
            concepts_concepts['GraphScore'] = (2 - concepts_concepts['GraphScore']) * concepts_concepts['GraphScore']

        # Add GraphScore column to results
        #   NOTE: We purposely perform an inner merge as an extra filter to exclude concepts which don't have an edge
        #   to any of the other concepts in the results, as they are very often noise. This could be an issue
        #   for very short texts, but this is rarely the case and even in that case it is never dramatic.
        #   Notice that if absolutely no edges are present among the concepts in the results,
        #   we never reach this point as we fall back above and return scores of 1
        results = pd.merge(results, concepts_concepts, how='inner', on='PageID')

        return results
