import pandas as pd

from db_cache_manager.db import DB

from graphai.core.common.config import config


class Ontology:
    def __init__(self):
        # Flag to know whether we've already loaded the data from the database
        self.loaded = False

        # Object of type pd.DataFrame with columns ['CategoryID', 'CategoryName'] holding the categories
        self.categories = None

        ################################################

        # Object of type pd.DataFrame with columns ['ChildCategoryID', 'ParentCategoryID'] holding the
        # category-category (parent-child) edges
        self.categories_categories = None

        ################################################

        # Object of type pd.DataFrame with columns ['PageID', 'CategoryID'] holding the concept-category edges
        self.concepts_categories = None

        ################################################

        # We can optionally set a reference to a Graph object to use the concept-concept edges,
        # but beware of doing this because the graph object takes several GB of memory space.
        # This object is however needed in the add_ontology_scores function.
        self.graph = None

    def fetch_from_db(self):
        if self.loaded:
            print('Ontology already loaded')
            print(self.loaded)
            return

        print('Actually loading the ontology tables...')

        db = DB(config['database'])

        ################################################

        use_new_tables = db.check_if_table_exists('francisco', 'Nodes_N_Category')
        if use_new_tables:
            print('Loading tables from "francisco" schema...')
        else:
            print('Loading tables from "graphontology" schema...')

        # Fetch category nodes
        print('Loading category nodes table...', end=' ')

        if use_new_tables:
            table_name = 'francisco.Nodes_N_Category'
        else:
            table_name = 'graphontology.Hierarchical_Cluster_Names_HighLevel'
        fields = ['CategoryID', 'CategoryName']
        self.categories = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

        print('Done')

        ################################################

        # Fetch category-category edges
        print('Loading category-category edges table...', end=' ')

        if use_new_tables:
            table_name = 'francisco.Edges_N_Category_N_Category'
        else:
            table_name = 'graphontology.Predefined_Knowledge_Tree_Hierarchy'
        fields = ['ChildCategoryID', 'ParentCategoryID']
        self.categories_categories = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

        print('Done')

        ################################################

        # Fetch concept-category edges
        print('Loading category-concept edges table...', end=' ')

        if use_new_tables:
            table_name = 'francisco.Edges_N_Category_N_Concept_T_OnlyDepth4'
        else:
            table_name = 'graphontology.Hierarchical_Clusters_Main'
        fields = ['PageID', 'CategoryID']
        self.concepts_categories = pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID', 'CategoryID'])

        print('Done')

        # Set the flag to avoid future reloads
        print('Setting ontology as loaded...')
        self.loaded = True

    def filter_concepts(self, results):
        self.fetch_from_db()
        return results[results['PageID'].isin(self.concepts_categories['PageID'])]

    def add_ontology_scores(self, results, smoothing=True):
        """
        Computes OntologyLocalScore and OntologyGlobalScore for the provided intermediate wikify results.

        Args:
            results (pd.DataFrame): A pandas DataFrame including the columns ['Keywords', 'PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore'].
            smoothing (bool): Whether to apply a transformation to the ontology scores that pushes scores away from 0.5. Default: True.

        Returns:
            pd.DataFrame: A pandas DataFrame with the original columns plus ['OntologyLocalScore', 'OntologyGlobalScore'].
        """

        self.fetch_from_db()

        # Extract local concepts and keep only edges between them
        concepts = results['PageID'].drop_duplicates()
        concepts_concepts = self.graph.concepts_concepts[
            self.graph.concepts_concepts['SourcePageID'].isin(concepts)
            & self.graph.concepts_concepts['TargetPageID'].isin(concepts)
        ]

        ################################################################

        # Build DataFrame with ontology concepts and their categories
        ontology_concepts_categories = pd.merge(concepts, self.concepts_categories, how='inner', on='PageID')

        ontology_concepts_categories = pd.merge(
            ontology_concepts_categories,
            self.categories_categories.rename(columns={'ChildCategoryID': 'CategoryID', 'ParentCategoryID': 'Category2ID'}),
            how='inner',
            on='CategoryID'
        )

        # Split concepts depending on whether they belong to the ontology
        ontology_concepts = ontology_concepts_categories['PageID']
        non_ontology_concepts = concepts[~concepts.isin(ontology_concepts)]

        # Fallback: Unlikely case when no concept in the results belongs to the ontology
        if len(ontology_concepts) == 0:
            results['OntologyLocalScore'] = 1
            results['OntologyGlobalScore'] = 1
            return results

        ################################################################

        # Build DataFrame with the non-ontology concepts and their neighbours
        non_ontology_concepts_concepts = pd.merge(
            non_ontology_concepts,
            concepts_concepts.rename(columns={'SourcePageID': 'PageID'}),
            how='inner',
            on='PageID'
        )

        # Build DataFrame with the non-ontology concepts and the categories of their ontology neighbours
        non_ontology_concepts_categories = pd.merge(
            non_ontology_concepts_concepts,
            ontology_concepts_categories.rename(columns={'PageID': 'TargetPageID'}),
            how='inner',
            on='TargetPageID'
        )

        # Add Coef column representing the relevance proportion of each neighbour, according to the edge NormalisedScore
        #   Notice that Coef sums to 1 for every PageID
        non_ontology_concepts_categories = pd.merge(
            non_ontology_concepts_categories,
            non_ontology_concepts_categories.groupby('PageID').aggregate(SumScore=('NormalisedScore', 'sum')).reset_index(),
            how='inner',
            on='PageID'
        )
        non_ontology_concepts_categories['Coef'] = non_ontology_concepts_categories['NormalisedScore'] / non_ontology_concepts_categories['SumScore']

        # Drop TargetPageID grouping by PageID and Categories and adding up the Coef
        #   Notice that Coef should still sum to 1 for every PageID
        non_ontology_concepts_categories = non_ontology_concepts_categories.groupby(by=['PageID', 'CategoryID', 'Category2ID']).aggregate(Coef=('Coef', 'sum')).reset_index()

        ################################################################

        # Concatenate both DataFrames for ontology and non-ontology concepts
        ontology_concepts_categories['Coef'] = 1
        concepts_categories = pd.concat([ontology_concepts_categories, non_ontology_concepts_categories])

        # Recover Keywords from results so that we can use it for the local counts
        concepts_categories = pd.merge(results[['Keywords', 'PageID']], concepts_categories, how='inner', on='PageID')

        ################################################################

        # Add local count: number of pages with the same keywords
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['Keywords']).aggregate(LocalCount=('Coef', 'sum')).reset_index(),
            how='left',
            on=['Keywords']
        )

        # Add local category count: number of pages among those with the same keywords sharing the same category
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['Keywords', 'CategoryID']).aggregate(Category1LocalCount=('Coef', 'sum')).reset_index(),
            how='left',
            on=['Keywords', 'CategoryID']
        )

        # Add local category2 count: number of pages among those with the same keywords sharing the same category2
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['Keywords', 'Category2ID']).aggregate(Category2LocalCount=('Coef', 'sum')).reset_index(),
            how='left',
            on=['Keywords', 'Category2ID']
        )

        # Add global category count: number of pages among all sharing the same category
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['CategoryID']).aggregate(Category1GlobalCount=('Coef', 'sum')).reset_index(),
            how='left',
            on=['CategoryID']
        )

        # Add global category2 count: number of pages among all sharing the same category2
        concepts_categories = pd.merge(
            concepts_categories,
            concepts_categories.groupby(by=['Category2ID']).aggregate(Category2GlobalCount=('Coef', 'sum')).reset_index(),
            how='left',
            on=['Category2ID']
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
        global_count = concepts_categories['Coef'].sum()
        if smoothing:
            concepts_categories['Ontology1LocalScore'] = h(concepts_categories['Category1LocalCount'], N=concepts_categories['LocalCount'], alpha=3, deg=2)
            concepts_categories['Ontology2LocalScore'] = h(concepts_categories['Category2LocalCount'], N=concepts_categories['LocalCount'], alpha=3, deg=2)
            concepts_categories['Ontology1GlobalScore'] = h(concepts_categories['Category1GlobalCount'], N=global_count, alpha=3, deg=8)
            concepts_categories['Ontology2GlobalScore'] = h(concepts_categories['Category2GlobalCount'], N=global_count, alpha=3, deg=8)
        else:
            concepts_categories['Ontology1LocalScore'] = concepts_categories['Category1LocalCount'] / concepts_categories['LocalCount']
            concepts_categories['Ontology2LocalScore'] = concepts_categories['Category2LocalCount'] / concepts_categories['LocalCount']
            concepts_categories['Ontology1GlobalScore'] = concepts_categories['Category1GlobalCount'] / global_count
            concepts_categories['Ontology2GlobalScore'] = concepts_categories['Category2GlobalCount'] / global_count

        # Normalise scores
        concepts_categories['Ontology1LocalScore'] = concepts_categories['Ontology1LocalScore'] / concepts_categories['Ontology1LocalScore'].max()
        concepts_categories['Ontology2LocalScore'] = concepts_categories['Ontology2LocalScore'] / concepts_categories['Ontology2LocalScore'].max()
        concepts_categories['Ontology1GlobalScore'] = concepts_categories['Ontology1GlobalScore'] / concepts_categories['Ontology1GlobalScore'].max()
        concepts_categories['Ontology2GlobalScore'] = concepts_categories['Ontology2GlobalScore'] / concepts_categories['Ontology2GlobalScore'].max()

        # Combine scores for 1 and 2 categories
        concepts_categories['OntologyLocalScore'] = 0.75 * concepts_categories['Ontology1LocalScore'] + 0.25 * concepts_categories['Ontology2LocalScore']
        concepts_categories['OntologyGlobalScore'] = 0.75 * concepts_categories['Ontology1GlobalScore'] + 0.25 * concepts_categories['Ontology2GlobalScore']

        # Keep only relevant columns
        concepts_categories = concepts_categories[['Keywords', 'PageID', 'Coef', 'OntologyLocalScore', 'OntologyGlobalScore']]

        ################################################################

        # Aggregate over Keywords and PageID and compute the convex combination of scores according to Coef
        concepts_categories['OntologyLocalScore'] = concepts_categories['Coef'] * concepts_categories['OntologyLocalScore']
        concepts_categories['OntologyGlobalScore'] = concepts_categories['Coef'] * concepts_categories['OntologyGlobalScore']
        concepts_categories = concepts_categories.groupby(by=['Keywords', 'PageID']).aggregate({'OntologyLocalScore': 'sum', 'OntologyGlobalScore': 'sum'}).reset_index()

        ################################################################

        # Merge results with final scores
        results = pd.merge(results, concepts_categories[['Keywords', 'PageID', 'OntologyLocalScore', 'OntologyGlobalScore']], how='inner', on=['Keywords', 'PageID'])

        return results
