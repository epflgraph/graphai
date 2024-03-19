import pandas as pd

from db_cache_manager.db import DB

from graphai.core.common.config import config


class NewConceptsGraph:
    def __init__(self):
        # Flag to know whether we've already loaded the data from the database
        self.loaded = False

        # DataFrames that hold all data in memory
        self.concepts = None                # Concept nodes
        self.concepts_concepts = None       # Concept-Concept edges (graph)
        self.categories = None              # Category nodes
        self.categories_categories = None   # Category-Category edges (ontology category tree)
        self.concepts_categories = None     # Concept-Category edges (ontology concept to level 4 category association)

    def load_from_db(self):
        # Do nothing if data already loaded
        if self.loaded:
            return

        ################################################################

        print('Actually loading the graph and ontology tables...')

        db = DB(config['database'])

        ################################################################

        # Load Concept nodes
        table_name = 'graph_ontology.Nodes_N_Concept'
        fields = ['id', 'name']
        columns = ['concept_id', 'concept_name']
        conditions = {'is_unused': False}
        self.concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)

        print('Concept nodes loaded')
        print(self.concepts.info())

        ################################################################

        # Load Concept-Concept edges (graph)
        table_name = 'graph_ontology.Edges_N_Concept_N_Concept_T_Undirected'
        fields = ['from_id', 'to_id', 'normalised_score']
        columns = ['source_concept_id', 'target_concept_id', 'score']
        self.concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=columns)

        # Complete edges with reversed ones
        self.concepts_concepts = pd.concat([
            self.concepts_concepts,
            self.concepts_concepts.rename(columns={'source_concept_id': 'target_concept_id', 'target_concept_id': 'source_concept_id'})
        ]).reset_index(drop=True)

        print('Concept-Concept edges loaded')
        print(self.concepts_concepts.info())

        ################################################################

        # Load Category nodes
        table_name = 'graph_ontology.Nodes_N_Category'
        fields = ['id', 'name', 'depth']
        columns = ['category_id', 'category_name', 'depth']
        self.categories = pd.DataFrame(db.find(table_name, fields=fields), columns=columns)

        print('Category nodes loaded')
        print(self.categories.info())

        ################################################################

        # Load Category-Category edges (ontology category tree)
        table_name = 'graph_ontology.Edges_N_Category_N_Category_T_ChildToParent'
        fields = ['from_id', 'to_id']
        columns = ['child_category_id', 'parent_category_id']
        self.categories_categories = pd.DataFrame(db.find(table_name, fields=fields), columns=columns)

        print('Category-Category edges loaded')
        print(self.categories_categories.info())

        ################################################################

        # Concept-Category edges (ontology concept to level 4 category association)
        #   To achieve this, we need to go through from concept to concept clusters,
        #   and from concept clusters to level 4 categories.

        # Load Concept-Cluster edges
        table_name = 'graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild'
        fields = ['to_id', 'from_id']
        columns = ['concept_id', 'cluster_id']
        concepts_clusters = pd.DataFrame(db.find(table_name, fields=fields), columns=columns)

        # Load Cluster-Category edges
        table_name = 'graph_ontology.Edges_N_Category_N_ConceptsCluster_T_ParentToChild'
        fields = ['to_id', 'from_id']
        columns = ['cluster_id', 'category_id']
        clusters_categories = pd.DataFrame(db.find(table_name, fields=fields), columns=columns)

        # Merge both tables through clusters to get the Concept-Category edges (ontology concept to level 4 category association)
        self.concepts_categories = pd.merge(concepts_clusters, clusters_categories, how='inner', on='cluster_id')[['concept_id', 'category_id']]

        print('Concept-Category edges loaded')
        print(self.concepts_categories.info())

        ################################################################

        # Set the loaded flag to True to avoid reloading again
        self.loaded = True

    def get_ontology_concepts(self):
        # Ensure data is loaded
        self.load_from_db()

        return self.concepts_categories['concept_id']

    def add_graph_score(self, results, smoothing=True):
        """
        Computes `graph_score` for the provided DataFrame containing concepts.

        Args:
            results (pd.DataFrame): A pandas DataFrame including the column ['concept_id'].
            smoothing (bool): Whether to apply a transformation to the `graph_score` that bumps scores to avoid
            a negative exponential shape. Default: True.

        Returns:
            pd.DataFrame: A pandas DataFrame with the original columns plus ['graph_score'].
        """

        # Ensure data is loaded
        self.load_from_db()

        # Extract concept ids from set of results
        concept_ids = list(results['concept_id'].drop_duplicates())

        # Keep only edges whose vertices are in the results' concepts
        concepts_concepts = self.concepts_concepts[
            self.concepts_concepts['source_concept_id'].isin(concept_ids)
            & self.concepts_concepts['target_concept_id'].isin(concept_ids)
        ]

        # Fallback: Unlikely case when no pair of concepts in the results share an edge, so induced graph are isolated vertices
        if len(concepts_concepts) == 0:
            results['graph_score'] = 1
            return results

        # Aggregate by source_concept_id adding up the scores (concepts_concepts contains edges and their reversed edges)
        concepts_concepts = concepts_concepts.groupby(by='source_concept_id').aggregate(graph_score=('score', 'sum')).reset_index()
        concepts_concepts = concepts_concepts.rename(columns={'source_concept_id': 'concept_id'})

        # Normalise so that scores are in [0, 1]
        concepts_concepts['graph_score'] = concepts_concepts['graph_score'] / concepts_concepts['graph_score'].max()

        # Smooth score if needed using the function f(x) = (2 - x) * x, bumping lower values to avoid very low scores
        if smoothing:
            concepts_concepts['graph_score'] = (2 - concepts_concepts['graph_score']) * concepts_concepts['graph_score']

        # Add graph_score column to results
        #   NOTE: We purposely perform an inner merge as an extra filter to exclude concepts which don't have an edge
        #   to any of the other concepts in the results, as they are very often noise. This could be an issue
        #   for very short texts, but this is rarely the case and even in that case it is never dramatic.
        #   Notice that if absolutely no edges are present among the concepts in the results,
        #   we never reach this point as we fall back above and return scores of 1
        results = pd.merge(results, concepts_concepts, how='inner', on='concept_id')

        return results

    def add_ontology_scores(self, results, smoothing=True):
        """
        Computes `ontology_local_score` and `ontology_global_score` for the provided DataFrame containing keywords and concepts.

        Args:
            results (pd.DataFrame): A pandas DataFrame including the columns ['keywords', 'concept_id'].
            smoothing (bool): Whether to apply a transformation to the ontology scores that pushes scores away from 0.5. Default: True.

        Returns:
            pd.DataFrame: A pandas DataFrame with the original columns plus ['ontology_local_score', 'ontology_global_score'].
        """

        self.load_from_db()

        # Extract local concepts and keep only edges between them
        concepts = results['concept_id'].drop_duplicates()
        concepts_concepts = self.concepts_concepts[
            self.concepts_concepts['source_concept_id'].isin(concepts)
            & self.concepts_concepts['target_concept_id'].isin(concepts)
        ]

        ################################################################

        # Build DataFrame with ontology concepts and their level 4 categories
        ontology_concepts_categories = pd.merge(concepts, self.concepts_categories, how='inner', on='concept_id')

        # Add column with level 3 category
        ontology_concepts_categories = pd.merge(
            ontology_concepts_categories.rename(columns={'category_id': 'level_4_category_id'}),
            self.categories_categories.rename(columns={'child_category_id': 'level_4_category_id', 'parent_category_id': 'level_3_category_id'}),
            how='inner',
            on='level_4_category_id'
        )

        # Split concepts depending on whether they belong to the ontology
        ontology_concepts = ontology_concepts_categories['concept_id']
        non_ontology_concepts = concepts[~concepts.isin(ontology_concepts)]

        # Fallback: Unlikely case when no concept in the results belongs to the ontology
        if len(ontology_concepts) == 0:
            results['ontology_local_score'] = 1
            results['ontology_global_score'] = 1
            return results

        ################################################################

        # Build DataFrame with the non-ontology concepts and their neighbours
        non_ontology_concepts_concepts = pd.merge(
            non_ontology_concepts,
            concepts_concepts.rename(columns={'source_concept_id': 'concept_id'}),
            how='inner',
            on='concept_id'
        )

        # Build DataFrame with the non-ontology concepts and the categories of their ontology neighbours
        non_ontology_concepts_categories = pd.merge(
            non_ontology_concepts_concepts,
            ontology_concepts_categories.rename(columns={'concept_id': 'target_concept_id'}),
            how='inner',
            on='target_concept_id'
        )

        # Add coef column representing the relevance proportion of each neighbour, according to the edge score
        #   Notice that coef sums to 1 for every concept_id
        non_ontology_concepts_categories = pd.merge(
            non_ontology_concepts_categories,
            non_ontology_concepts_categories.groupby('concept_id').aggregate(sum_score=('score', 'sum')).reset_index(),
            how='inner',
            on='concept_id'
        )
        non_ontology_concepts_categories['coef'] = non_ontology_concepts_categories['score'] / non_ontology_concepts_categories['sum_score']

        # Drop target_concept_id grouping by concept_id and both categories and adding up the coef
        #   Notice that coef should still sum to 1 for every concept_id
        non_ontology_concepts_categories = non_ontology_concepts_categories.groupby(by=['concept_id', 'level_3_category_id', 'level_4_category_id']).aggregate(coef=('coef', 'sum')).reset_index()

        ################################################################

        # Concatenate both DataFrames for ontology and non-ontology concepts
        ontology_concepts_categories['coef'] = 1
        concepts_categories = pd.concat([ontology_concepts_categories, non_ontology_concepts_categories])

        # Recover keywords from results so that we can use it for the local counts
        concepts_categories = pd.merge(results[['keywords', 'concept_id']], concepts_categories, how='inner', on='concept_id')

        ################################################################

        # Add local_count: number of pages with the same keywords
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['keywords']).aggregate(local_count=('coef', 'sum')).reset_index(),
            how='left',
            on=['keywords']
        )

        # Add level_4_category_local_count: number of pages among those with the same keywords sharing the same level 4 category
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['keywords', 'level_4_category_id']).aggregate(level_4_category_local_count=('coef', 'sum')).reset_index(),
            how='left',
            on=['keywords', 'level_4_category_id']
        )

        # Add level_3_category_local_count: number of pages among those with the same keywords sharing the same level 3 category
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['keywords', 'level_3_category_id']).aggregate(level_3_category_local_count=('coef', 'sum')).reset_index(),
            how='left',
            on=['keywords', 'level_3_category_id']
        )

        # Add level_4_category_global_count: number of pages among all sharing the same level 4 category
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['level_4_category_id']).aggregate(level_4_category_global_count=('coef', 'sum')).reset_index(),
            how='left',
            on=['level_4_category_id']
        )

        # Add level_3_category_global_count: number of pages among all sharing the same level 3 category
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['level_3_category_id']).aggregate(level_3_category_global_count=('coef', 'sum')).reset_index(),
            how='left',
            on=['level_3_category_id']
        )

        ################################################################

        # S-shaped function h: [0, N] -> [0, 1] such that
        #   * h is strictly increasing
        #   * h(0) = 0 and h(N) = 1
        #   * h is convex in (1, alpha) and concave in (alpha, N), for alpha in (0, N).
        #   * h(alpha) = alpha
        #   * h differentiable in alpha
        #   * h branches are polynomials of degree deg
        def h(x, N, alpha, deg):
            # Make everything a pd.Series
            N = pd.Series(N, index=range(len(x)))
            alpha = pd.Series(alpha, index=range(len(x)))

            # Make sure alpha is in (0, N)
            alpha = alpha.clip(lower=0.5, upper=N - 0.5)

            def f(t):
                return (1 / ((alpha / N)**(deg - 1))) * (t / N)**deg

            def g(t):
                return 1 - (1 / (1 - (alpha / N))**(deg - 1)) * (1 - t / N)**deg

            indicator = (x <= alpha).astype(int)
            return indicator * f(x) + (1 - indicator) * g(x)

        # Compute scores
        global_count = concepts_categories['coef'].sum()
        if smoothing:
            concepts_categories['level_4_ontology_local_score'] = h(concepts_categories['level_4_category_local_count'], N=concepts_categories['local_count'], alpha=3, deg=2)
            concepts_categories['level_3_ontology_local_score'] = h(concepts_categories['level_3_category_local_count'], N=concepts_categories['local_count'], alpha=3, deg=2)
            concepts_categories['level_4_ontology_global_score'] = h(concepts_categories['level_4_category_global_count'], N=global_count, alpha=3, deg=8)
            concepts_categories['level_3_ontology_global_score'] = h(concepts_categories['level_3_category_global_count'], N=global_count, alpha=3, deg=8)
        else:
            concepts_categories['level_4_ontology_local_score'] = concepts_categories['level_4_category_local_count'] / concepts_categories['local_count']
            concepts_categories['level_3_ontology_local_score'] = concepts_categories['level_3_category_local_count'] / concepts_categories['local_count']
            concepts_categories['level_4_ontology_global_score'] = concepts_categories['level_4_category_global_count'] / global_count
            concepts_categories['level_3_ontology_global_score'] = concepts_categories['level_3_category_global_count'] / global_count

        # Normalise scores
        concepts_categories['level_4_ontology_local_score'] = concepts_categories['level_4_ontology_local_score'] / concepts_categories['level_4_ontology_local_score'].max()
        concepts_categories['level_3_ontology_local_score'] = concepts_categories['level_3_ontology_local_score'] / concepts_categories['level_3_ontology_local_score'].max()
        concepts_categories['level_4_ontology_global_score'] = concepts_categories['level_4_ontology_global_score'] / concepts_categories['level_4_ontology_global_score'].max()
        concepts_categories['level_3_ontology_global_score'] = concepts_categories['level_3_ontology_global_score'] / concepts_categories['level_3_ontology_global_score'].max()

        # Combine scores for level 4 and 3 categories as a convex combination, with weights of 0.75 and 0.25, respectively
        concepts_categories['ontology_local_score'] = 0.75 * concepts_categories['level_4_ontology_local_score'] + 0.25 * concepts_categories['level_3_ontology_local_score']
        concepts_categories['ontology_global_score'] = 0.75 * concepts_categories['level_4_ontology_global_score'] + 0.25 * concepts_categories['level_3_ontology_global_score']

        # Keep only relevant columns
        concepts_categories = concepts_categories[['keywords', 'concept_id', 'coef', 'ontology_local_score', 'ontology_global_score']]

        ################################################################

        # Aggregate over keywords and concept_id and compute the convex combination of scores according to coef
        concepts_categories['ontology_local_score'] = concepts_categories['coef'] * concepts_categories['ontology_local_score']
        concepts_categories['ontology_global_score'] = concepts_categories['coef'] * concepts_categories['ontology_global_score']
        concepts_categories = concepts_categories.groupby(by=['keywords', 'concept_id']).aggregate({'ontology_local_score': 'sum', 'ontology_global_score': 'sum'}).reset_index()

        ################################################################

        # Merge results with final scores
        results = pd.merge(results, concepts_categories[['keywords', 'concept_id', 'ontology_local_score', 'ontology_global_score']], how='inner', on=['keywords', 'concept_id'])

        return results
