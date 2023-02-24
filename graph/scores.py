import numpy as np
import pandas as pd

from interfaces.db import DB


class ConceptsGraph:
    def __init__(self):
        db = DB()

        print('A')

        table_name = 'graph.Nodes_N_Concept'
        fields = ['PageID', 'PageTitle']
        self.concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
        concept_ids = list(self.concepts['PageID'])

        print('B')

        table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
        fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
        conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
        self.concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

        print('C')

        self.concepts_concepts = pd.concat([
            self.concepts_concepts,
            self.concepts_concepts.rename(columns={'SourcePageID': 'TargetPageID', 'TargetPageID': 'SourcePageID'})
        ]).reset_index(drop=True)

        print('D')

        # Store successor and predecessor sets as attributes
        self.successors = self.concepts_concepts.groupby(by='SourcePageID').aggregate({'TargetPageID': set})
        self.predecessors = self.concepts_concepts.groupby(by='TargetPageID').aggregate({'SourcePageID': set})
        self.sources = set(self.successors.index)
        self.targets = set(self.predecessors.index)

        print('E')

    def compute_scores(self, source_page_ids, target_page_ids):
        """
        Computes the graph scores for all possible source-target pairs from two lists of page ids.
        The graph score of a pair (s, t) is computed as:

        :math:`\\displaystyle score(s, t) = 1 - \\frac{1}{1 + \\ln(1 + rebound(s, t) * \\ln(out(s, t)))}`, with

        :math:`\\displaystyle rebound(s, t) = \\frac{1 + \\min\\{in(s, t), out(s, t)\\}}{1 + \max\\{in(s, t), out(s, t)\\}}`,

        :math:`in(s, t) =` number of paths from t to s,

        :math:`out(s, t) =` number of paths from s to t.

        Args:
            source_page_ids (list[int]): List of source page ids.
            target_page_ids (list[int]): List of target page ids.

        Returns:
            list[dict[str]]: A list with all possible source-target pairs and their graph score. Each element of the list
            has keys 'source_page_id' (int), 'target_page_id' (int) and 'score' (float).

        Examples:
            >>> cg = ConceptsGraph()
            >>> cg.compute_scores([6220, 1196], [18973446, 9417, 946975])
            [{'source_page_id': 6220, 'target_page_id': 18973446, 'score': 0.5193576613841849}, {'source_page_id': 6220, 'target_page_id': 9417, 'score': 0.4740698357575134}, {'source_page_id': 6220, 'target_page_id': 946975, 'score': 0.3591100561928845}, {'source_page_id': 1196, 'target_page_id': 18973446, 'score': 0.5343247664351338}, {'source_page_id': 1196, 'target_page_id': 9417, 'score': 0.530184349205493}, {'source_page_id': 1196, 'target_page_id': 946975, 'score': 0.40043881227279876}]
        """
        pairs = [(s, t) for s in source_page_ids for t in target_page_ids]

        results = []
        for s, t in pairs:

            # If both pages are the same, then the score is 1
            if s == t:
                results.append({
                    'source_page_id': s,
                    'target_page_id': t,
                    'score': 1
                })
                continue

            # If s has no successors or t has no predecessors, then there are no outgoing paths. The score is 0.
            if (s not in self.sources) or (t not in self.targets):
                results.append({
                    'source_page_id': s,
                    'target_page_id': t,
                    'score': 0
                })
                continue

            # Compute the number of outgoing paths of length <= 2
            s_out = set(self.successors.loc[s, 'TargetPageID']) - {s}
            t_in = set(self.predecessors.loc[t, 'SourcePageID']) - {t}
            n_out_paths = len(s_out & t_in) + (1 if t in s_out else 0)

            # If there are no outgoing paths, the score is zero
            if n_out_paths == 0:
                results.append({
                    'source_page_id': s,
                    'target_page_id': t,
                    'score': 0
                })
                continue

            # Compute the number of ingoing paths of length <= 2
            if (s not in self.targets) or (t not in self.sources):
                # If s has no predecessors or t has no successors, then there are no ingoing paths.
                n_in_paths = 0
            else:
                s_in = set(self.predecessors.loc[s, 'SourcePageID']) - {s}
                t_out = set(self.successors.loc[s, 'TargetPageID']) - {t}
                n_in_paths = len(s_in & t_out) + (1 if s in t_out else 0)

            # Compute score
            rebound = (1 + min(n_out_paths, n_in_paths)) / (1 + max(n_out_paths, n_in_paths))
            score = 1 - 1 / (1 + np.log(1 + rebound * np.log(n_out_paths)))

            # Append result
            results.append({
                'source_page_id': s,
                'target_page_id': t,
                'score': score
            })

        return results

    def add_graph_score(self, results):
        concept_ids = list(results['PageID'].drop_duplicates())
        concepts_concepts = self.concepts_concepts[
            self.concepts_concepts['SourcePageID'].isin(concept_ids) &
            self.concepts_concepts['TargetPageID'].isin(concept_ids)
        ]
        concepts_concepts = concepts_concepts.groupby(by='SourcePageID').aggregate(GraphScore=('NormalisedScore', 'sum')).reset_index()
        concepts_concepts['GraphScore'] = concepts_concepts['GraphScore'] / concepts_concepts['GraphScore'].max()
        concepts_concepts = concepts_concepts.rename(columns={'SourcePageID': 'PageID'})

        results = pd.merge(results, concepts_concepts, how='inner', on='PageID')

        return results


# Function g: [1, N] -> [0, 1] satisfying the following:
#   g(1) = 0
#   g(N) = 1
#   g increasing
#   g concave
def g(x, N):
    return np.sin((np.pi / 2) * (x - 1) / (N - 1))


class Ontology:
    def __init__(self):
        db = DB()

        ################################################

        # Fetch category nodes
        table_name = 'graphontology.Hierarchical_Cluster_Names_HighLevel'
        fields = ['CategoryID', 'CategoryName']
        self.categories = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

        # Extract category ids for faster access
        self.category_ids = set(self.categories['CategoryID'])

        # Store category names indexing by CategoryID for faster access
        self.category_names = self.categories.set_index('CategoryID')['CategoryName']

        ################################################

        # Fetch category-category edges
        table_name = 'graphontology.Predefined_Knowledge_Tree_Hierarchy'
        fields = ['ChildCategoryID', 'ParentCategoryID']
        self.categories_categories = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

        # Extract child category ids for faster access
        self.category_child_ids = set(self.categories_categories['ChildCategoryID'])

        # Store category parents indexing by ChildCategoryID for faster access
        self.category_parents = self.categories_categories.set_index('ChildCategoryID')['ParentCategoryID']

        ################################################

        # Fetch concept-category edges
        table_name = 'graphontology.Hierarchical_Clusters_Main'
        fields = ['PageID', 'CategoryID']
        self.concepts_categories = pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID', 'CategoryID'])

        # Extract concept and category ids for faster access
        self.concept_ids = set(self.concepts_categories['PageID'])

        # Store category ids indexing by PageID for faster access
        self.concept_categories = self.concepts_categories.set_index('PageID')['CategoryID']

    def get_concept_category(self, page_id):
        if page_id not in self.concept_ids:
            return None

        return self.concept_categories.at[page_id]

    def add_concepts_category(self, results):
        return pd.merge(results, self.concept_categories, how='inner', on='PageID')

    def add_categories_category(self, results):
        return pd.merge(
            results,
            self.categories_categories.rename(columns={'ChildCategoryID': 'CategoryID', 'ParentCategoryID': 'Category2ID'}),
            how='inner',
            on='CategoryID'
        )

    def filter_concepts(self, results):
        return results[results['PageID'].isin(self.concept_ids)]

    def add_ontology_score(self, results):
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
            results.groupby(by=['Keywords', 'CategoryID']).aggregate(CategoryLocalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['Keywords', 'CategoryID']
        )

        # Add global category count: number of pages among all sharing the same category
        results = pd.merge(
            results,
            results.groupby(by=['CategoryID']).aggregate(CategoryGlobalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['CategoryID']
        )

        # Add local category2 count: number of pages among those with the same keywords sharing the same category2
        results = pd.merge(
            results,
            results.groupby(by=['Keywords', 'Category2ID']).aggregate(Category2LocalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['Keywords', 'Category2ID']
        )

        # Add global category2 count: number of pages among all sharing the same category2
        results = pd.merge(
            results,
            results.groupby(by=['Category2ID']).aggregate(Category2GlobalCount=('PageID', 'count')).reset_index(),
            how='left',
            on=['Category2ID']
        )

        # Compute scores
        results['OntologyLocalScore'] = results['CategoryLocalCount'] / results['LocalCount']
        results['OntologyGlobalScore'] = results['CategoryGlobalCount'] / len(results)
        results['Ontology2LocalScore'] = results['Category2LocalCount'] / results['LocalCount']
        results['Ontology2GlobalScore'] = results['Category2GlobalCount'] / len(results)

        # Normalise scores
        results['OntologyLocalScore'] = results['OntologyLocalScore'] / results['OntologyLocalScore'].max()
        results['OntologyGlobalScore'] = results['OntologyGlobalScore'] / results['OntologyGlobalScore'].max()
        results['Ontology2LocalScore'] = results['Ontology2LocalScore'] / results['Ontology2LocalScore'].max()
        results['Ontology2GlobalScore'] = results['Ontology2GlobalScore'] / results['Ontology2GlobalScore'].max()

        # Ontology score
        results['OntologyScore'] = results['OntologyGlobalScore']

        # Drop temporary columns
        results = results.drop(columns=['LocalCount', 'CategoryLocalCount', 'CategoryGlobalCount', 'Category2LocalCount', 'Category2GlobalCount'])

        pd.set_option('display.max_rows', 400)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        return results
