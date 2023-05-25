import pandas as pd

from graphai.core.interfaces.db import DB


class Ontology:
    def __init__(self):
        # Flag to know whether we've already loaded the data from the database
        self.loaded = False

        # Object of type pd.DataFrame with columns ['CategoryID', 'CategoryName'] holding the categories
        self.categories = None

        # Set containing the CategoryID of all categories
        self.category_ids = None

        # Object of type pd.Series containing 'CategoryName' indexed by 'CategoryID'
        self.category_names = None

        ################################################

        # Object of type pd.DataFrame with columns ['ChildCategoryID', 'ParentCategoryID'] holding the
        # category-category (parent-child) edges
        self.categories_categories = None

        # Set containing the CategoryID of all children categories
        self.category_child_ids = None

        # Object of type pd.Series containing 'ParentCategoryID' indexed by 'ChildCategoryID'
        self.category_parents = None

        ################################################

        # Object of type pd.DataFrame with columns ['PageID', 'CategoryID'] holding the concept-category edges
        self.concepts_categories = None

        # Set containing the PageID of all concepts that have a category
        self.concept_ids = None

        # Object of type pd.Series containing 'CategoryID' indexed by 'PageID'
        self.concept_categories = None

    def fetch_from_db(self):
        if self.loaded:
            print('Ontology already loaded')
            print(self.loaded)
            return

        print('Actually loading the ontology tables...')

        db = DB()

        ################################################

        # Fetch category nodes
        # table_name = 'graphontology.Hierarchical_Cluster_Names_HighLevel'
        table_name = 'francisco.Nodes_N_Category'
        fields = ['CategoryID', 'CategoryName']
        self.categories = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

        # Extract category ids for faster access
        self.category_ids = set(self.categories['CategoryID'])

        # Store category names indexing by CategoryID for faster access
        self.category_names = self.categories.set_index('CategoryID')['CategoryName']

        ################################################

        # Fetch category-category edges
        # table_name = 'graphontology.Predefined_Knowledge_Tree_Hierarchy'
        table_name = 'francisco.Edges_N_Category_N_Category'
        fields = ['ChildCategoryID', 'ParentCategoryID']
        self.categories_categories = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

        # Extract child category ids for faster access
        self.category_child_ids = set(self.categories_categories['ChildCategoryID'])

        # Store category parents indexing by ChildCategoryID for faster access
        self.category_parents = self.categories_categories.set_index('ChildCategoryID')['ParentCategoryID']

        ################################################

        # Fetch concept-category edges
        # table_name = 'graphontology.Hierarchical_Clusters_Main'
        table_name = 'francisco.Edges_N_Category_N_Concept_T_OnlyDepth4'
        fields = ['PageID', 'CategoryID']
        self.concepts_categories = pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID', 'CategoryID'])

        # Extract concept and category ids for faster access
        self.concept_ids = set(self.concepts_categories['PageID'])

        # Store category ids indexing by PageID for faster access
        self.concept_categories = self.concepts_categories.set_index('PageID')['CategoryID']

        # Setting the flag to avoid future reloads
        self.loaded = True

    def get_concept_category(self, page_id):
        self.fetch_from_db()
        if page_id not in self.concept_ids:
            return None
        return self.concept_categories.at[page_id]

    def get_predefined_tree(self):
        self.fetch_from_db()
        return self.categories_categories.to_dict(orient='records')

    def get_category_parent(self, category_id):
        self.fetch_from_db()
        if category_id not in self.category_parents.index:
            return None
        else:
            return [{
                'ParentCategoryID': self.category_parents[category_id],
                'ChildCategoryID': category_id
            }]

    def get_category_children(self, category_id):
        self.fetch_from_db()
        if category_id not in self.category_ids:
            return None
        else:
            cat_to_cat = self.categories_categories
            return cat_to_cat.loc[cat_to_cat['ParentCategoryID'] == category_id].to_dict(orient='records')

    def add_concepts_category(self, results):
        self.fetch_from_db()
        return pd.merge(results, self.concept_categories, how='inner', on='PageID')

    def add_categories_category(self, results):
        self.fetch_from_db()
        return pd.merge(
            results,
            self.categories_categories.rename(columns={'ChildCategoryID': 'CategoryID', 'ParentCategoryID': 'Category2ID'}),
            how='inner',
            on='CategoryID'
        )

    def filter_concepts(self, results):
        self.fetch_from_db()
        return results[results['PageID'].isin(self.concept_ids)]

    def add_ontology_scores(self, results, smoothing=True):
        self.fetch_from_db()
        # Add concepts category column
        results = pd.merge(results, self.concepts_categories, how='inner', on='PageID')

        # Add categories category column
        results = pd.merge(
            results,
            self.categories_categories.rename(columns={'ChildCategoryID': 'CategoryID', 'ParentCategoryID': 'Category2ID'}),
            how='inner',
            on='CategoryID'
        )

        # Add local count: number of pages with the same keywords
        results = pd.merge(
            results,
            results.groupby(by=['Keywords']).aggregate(LocalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['Keywords']
        )

        # Add local category count: number of pages among those with the same keywords sharing the same category
        results = pd.merge(
            results,
            results.groupby(by=['Keywords', 'CategoryID']).aggregate(Category1LocalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['Keywords', 'CategoryID']
        )

        # Add local category2 count: number of pages among those with the same keywords sharing the same category2
        results = pd.merge(
            results,
            results.groupby(by=['Keywords', 'Category2ID']).aggregate(Category2LocalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['Keywords', 'Category2ID']
        )

        # Add global category count: number of pages among all sharing the same category
        results = pd.merge(
            results,
            results.groupby(by=['CategoryID']).aggregate(Category1GlobalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['CategoryID']
        )

        # Add global category2 count: number of pages among all sharing the same category2
        results = pd.merge(
            results,
            results.groupby(by=['Category2ID']).aggregate(Category2GlobalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['Category2ID']
        )

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
        if smoothing:
            results['Ontology1LocalScore'] = h(results['Category1LocalCount'], N=results['LocalCount'], alpha=3, deg=2)
            results['Ontology2LocalScore'] = h(results['Category2LocalCount'], N=results['LocalCount'], alpha=3, deg=2)
            results['Ontology1GlobalScore'] = h(results['Category1GlobalCount'], N=len(results), alpha=3, deg=8)
            results['Ontology2GlobalScore'] = h(results['Category2GlobalCount'], N=len(results), alpha=3, deg=8)
        else:
            results['Ontology1LocalScore'] = results['Category1LocalCount'] / results['LocalCount']
            results['Ontology2LocalScore'] = results['Category2LocalCount'] / results['LocalCount']
            results['Ontology1GlobalScore'] = results['Category1GlobalCount'] / len(results)
            results['Ontology2GlobalScore'] = results['Category2GlobalCount'] / len(results)

        # Normalise scores
        results['Ontology1LocalScore'] = results['Ontology1LocalScore'] / results['Ontology1LocalScore'].max()
        results['Ontology2LocalScore'] = results['Ontology2LocalScore'] / results['Ontology2LocalScore'].max()
        results['Ontology1GlobalScore'] = results['Ontology1GlobalScore'] / results['Ontology1GlobalScore'].max()
        results['Ontology2GlobalScore'] = results['Ontology2GlobalScore'] / results['Ontology2GlobalScore'].max()

        # Combine scores for 1 and 2 categories
        results['OntologyLocalScore'] = 0.75 * results['Ontology1LocalScore'] + 0.25 * results['Ontology2LocalScore']
        results['OntologyGlobalScore'] = 0.75 * results['Ontology1GlobalScore'] + 0.25 * results['Ontology2GlobalScore']

        # Drop temporary columns
        results = results.drop(columns=['LocalCount', 'Category1LocalCount', 'Category2LocalCount', 'Category1GlobalCount', 'Category2GlobalCount', 'Ontology1LocalScore', 'Ontology2LocalScore', 'Ontology1GlobalScore', 'Ontology2GlobalScore'])

        return results
