import numpy as np
from db_cache_manager.db import DB
import pandas as pd
import random
from scipy.sparse import csr_array, csr_matrix, vstack, spmatrix

from graphai.core.common.common_utils import invert_dict
from graphai.core.common.config import config


def db_results_to_pandas_df(results, cols):
    return pd.DataFrame(results, columns=cols)


def get_id_dict(ids):
    ids_set = list(set(ids))
    return invert_dict(dict(enumerate(sorted(ids_set))))


def make_adj_undirected(graph_adj):
    """
    Makes a directed graph undirected by making the adjacency matrix symmetric
    :param graph_adj: Adjacency matrix
    :return: Undirected graph adjacency matrix
    """
    return graph_adj + graph_adj.transpose()


def derive_col_to_col_graph(orig_adj):
    """
    Derives the adjacency matrix of the graph induced on the columns of the original adjacency
    matrix through its rows.
    :param orig_adj: Original adjacency matrix
    :return: A^T.A
    """
    return orig_adj.transpose().dot(orig_adj)


def get_col_to_col_dict(df, source_col, target_col):
    """
    Gets a dictionary mapping the elements of one dataframe column to the elements of another
    :param df: The dataframe
    :param source_col: Source column (keys)
    :param target_col: Target column (values)
    :return: The dictionary
    """
    from_values = df[source_col].values.tolist()
    to_values = df[target_col].values.tolist()
    return {from_values[i]: to_values[i] for i in range(len(from_values))}


def return_chosen_indices(base_list, indices):
    return [base_list[i] for i in indices]


def remove_invalid_pairs(l_main, l_secondary_1, l_secondary_2, ref_dict):
    """
    Takes two lists that refer to the rows and columns of a matrix plus a reference dictionary,
    and only keeps those indices of the two lists whose elements in the "main" list appear in
    the reference dictionary. In other words, eliminates the row-col or col-row pairs whose
    row/col (respectively) does not appear in the reference dictionary.
    :param l_main: The main list, which will be checked against the dictionary
    :param l_secondary: The secondary list
    :param ref_dict: The reference dictionary
    :return: Two cleaned up lists in the order that they were provided in
    """
    valid_indices = [i for i in range(len(l_main)) if ref_dict.get(l_main[i], None) is not None]
    return (return_chosen_indices(l_main, valid_indices),
            return_chosen_indices(l_secondary_1, valid_indices),
            return_chosen_indices(l_secondary_2, valid_indices))


def create_graph_from_df(df, source_col, target_col, weight_col=None,
                         col_dict=None, row_dict=None, pool_rows_and_cols=False,
                         make_symmetric=False):
    """
    Takes a dataframe containing the edges for a graph, and turns it into a directed or undirected
    graph (represented by an adjacency matrix)
    :param df: The dataframe
    :param source_col: The column for the source nodes
    :param target_col: The column for the target nodes
    :param weight_col: The column for edge weights
    :param col_dict: Precomputed dictionary for the adj matrix columns (targets), optional
    :param row_dict: Precomputed dictionary for the adj matrix rows (sources), optional
    :param pool_rows_and_cols: Whether to pool together the rows and columns, used for Wikipedia concept
    :param make_symmetric: Whether to make the graph undirected
    :return: The adjacency matrix, row dictionary, and column dictionary (id to index)
    """
    if weight_col is not None:
        df = df.dropna(axis=0, how='all', subset=[weight_col], inplace=False)
    rows = df[source_col].values.tolist()
    cols = df[target_col].values.tolist()
    if weight_col is not None:
        data = df[weight_col].values.tolist()
    else:
        data = [1] * len(cols)
    if pool_rows_and_cols:
        rows_and_cols = rows + cols
        row_dict = get_id_dict(rows_and_cols)
        col_dict = row_dict
    else:
        if row_dict is None:
            row_dict = get_id_dict(rows)
        else:
            rows, cols, data = remove_invalid_pairs(rows, cols, data, row_dict)
        if col_dict is None:
            col_dict = get_id_dict(cols)
        else:
            cols, rows, data = remove_invalid_pairs(cols, rows, data, col_dict)
    row_inds = [row_dict[k] for k in rows]

    col_inds = [col_dict[k] for k in cols]

    print(len(data), len(row_inds), len(col_inds))

    graph_adj = csr_array((data, (row_inds, col_inds)), shape=(len(row_dict), len(col_dict)))
    graph_adj.eliminate_zeros()
    if make_symmetric:
        graph_adj = make_adj_undirected(graph_adj)

    return graph_adj, row_dict, col_dict


def convert_to_csr_matrix(g):
    """
    Converts a given matrix or ndarray to a CSR matrix
    :param g: Matrix or ndarray
    :return: CSR matrix
    """
    return csr_matrix(g)


def to_ndarray_and_flatten(a):
    if isinstance(a, np.ndarray):
        return np.array(a).flatten()
    elif isinstance(a, spmatrix):
        return np.array(a.todense()).flatten()
    else:
        return a


def compute_average(score, n, avg):
    assert avg in ['linear', 'log', 'none']

    if avg == 'none':
        return score

    score = to_ndarray_and_flatten(score)
    n = to_ndarray_and_flatten(n)
    if isinstance(n, np.ndarray):
        n[np.where(n == 0)[0]] = 1
    else:
        if n == 0:
            n = 1

    if avg == 'linear':
        score /= n
    elif avg == 'log':
        score /= np.log(1 + n)
    return score


def average_and_combine(s1, s2, l1, l2, avg, coeffs, skip_empty=False):
    assert coeffs is None or (all([c >= 0 for c in coeffs]) and sum(coeffs) > 0)
    if s2 is None or l2 is None:
        s2 = 0
        l2 = 0
    if coeffs is None:
        score = s1 + s2
        denominator = l1 + l2
        score = compute_average(score, denominator, avg)
    else:
        averages_1 = compute_average(s1, l1, avg)
        averages_2 = compute_average(s2, l2, avg)
        if skip_empty:
            if isinstance(averages_1, np.ndarray):
                all_lengths = [to_ndarray_and_flatten(l1), to_ndarray_and_flatten(l2)]
                coeff_sum = [sum([coeffs[j] for j in range(2) if all_lengths[j][i] > 0])
                             for i in range(averages_1.shape[0])]
                coeff_sum = np.array([x if x > 0 else 1 for x in coeff_sum])
            else:
                all_lengths = [l1, l2]
                coeff_sum = sum([coeffs[i] for i in range(2) if all_lengths[i] > 0])
        else:
            coeff_sum = sum(coeffs)
        score = (coeffs[0] * averages_1 + coeffs[1] * averages_2) / coeff_sum
    return score


class OntologyData:
    def __init__(self, test_mode=False, **kwargs):
        self.loaded = False
        self.db_config = None
        self.ontology_concept_names = None
        self.ontology_categories = None
        self.category_depth_dict = None
        self.non_ontology_concept_names = None
        self.concept_concept_graphscore = None
        self.ontology_and_anchor_concepts_id_to_index = None
        self.symmetric_concept_concept_matrix = dict()
        self.category_category = None
        self.category_category_dict = None
        self.category_concept = None
        self.category_cluster = None
        self.cluster_concept = None
        self.category_concept_dict = None
        self.category_cluster_dict = None
        self.cluster_concept_dict = None
        self.category_anchors_dict = None
        # Parameters for test mode
        self.test_mode = test_mode
        self.random_state = kwargs.get('random_state', 0)
        self.test_ratio = kwargs.get('test_ratio', 0.0)
        self.test_ids = None
        self.test_category_concept = None
        self.test_cluster_concept = None

    def load_data(self):
        if not self.loaded:
            self.db_config = config['database']
            # First loading concepts: ontology concepts, anchor concepts, and ontology-neighborhood concepts
            self.load_ontology_concept_names()
            self.load_ontology_categories()
            self.load_non_ontology_concept_names()
            # Now loading the concept-concept table and the matrix from neighborhood to ontology/anchors
            self.load_concept_concept_graphscore()
            self.compute_symmetric_concept_concept_matrix()
            # Now loading category-category and category-concept tables
            self.load_category_category()
            self.load_category_concept()
            # Now loading aggregated anchor concept lists for each category
            self.load_anchor_page_dict()
            # Finally, we compute aggregated matrices that map each concept to each category
            self.compute_precalculated_similarity_matrices()
            self.loaded = True

    def load_ontology_concept_names(self):
        db_manager = DB(self.db_config)
        self.ontology_concept_names = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT id, name FROM graph_ontology.Nodes_N_Concept WHERE is_ontology_concept=1"),
            ['id', 'name']
        )
        if self.test_mode:
            test_n = int(self.ontology_concept_names.shape[0] * self.test_ratio)
            if test_n > 0:
                all_ids = self.ontology_concept_names['id'].values.tolist()
                self.test_ids = random.sample(all_ids, test_n)
                self.ontology_concept_names = self.ontology_concept_names.loc[
                    self.ontology_concept_names['id'].apply(lambda x: x not in self.test_ids)
                ]


    def load_ontology_categories(self):
        db_manager = DB(self.db_config)
        self.ontology_categories = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT a.id AS category_id, a.depth AS depth, b.id AS concept_id, b.name AS concept_name "
            "FROM graph_ontology.Nodes_N_Category a JOIN graph_ontology.Nodes_N_Concept b "
            "ON a.reference_page_id=b.id "
            "WHERE b.is_ontology_category=1;"),
            ['category_id', 'depth', 'id', 'name']
        )
        cat_ids = self.ontology_categories.category_id.values.tolist()
        depths = self.ontology_categories.depth.values.tolist()
        self.category_depth_dict = {cat_ids[i]: depths[i] for i in range(len(cat_ids))}

    def load_non_ontology_concept_names(self):
        db_manager = DB(self.db_config)
        self.non_ontology_concept_names = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT id, name FROM graph_ontology.Nodes_N_Concept WHERE is_ontology_neighbour=1"),
            ['id', 'name']
        )

    def load_concept_concept_graphscore(self):
        db_manager = DB(self.db_config)
        self.concept_concept_graphscore = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT from_id, to_id, score FROM graph_ontology.Edges_N_Concept_N_Concept_T_Undirected"),
            ['from_id', 'to_id', 'score']
        )

    def load_category_category(self):
        db_manager = DB(self.db_config)
        self.category_category = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT from_id, to_id FROM graph_ontology.Edges_N_Category_N_Category_T_ChildToParent;"),
            ['from_id', 'to_id']
        )
        category_category_agg = self.category_category.assign(
            id=self.category_category.from_id.apply(lambda x: [x])
        )[['to_id', 'id']].groupby('to_id').agg(sum).reset_index()
        self.category_category_dict = get_col_to_col_dict(category_category_agg, 'to_id', 'id')

    def load_category_concept(self):
        db_manager = DB(self.db_config)
        self.category_cluster = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT from_id, to_id FROM graph_ontology.Edges_N_Category_N_ConceptsCluster_T_ParentToChild"),
            ['from_id', 'to_id']
        )
        self.cluster_concept = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT from_id, to_id FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild"),
            ['from_id', 'to_id']
        )

        self.category_concept = (
            pd.merge(self.category_cluster, self.cluster_concept,
                     left_on='to_id', right_on='from_id',
                     suffixes=('_cat', '_concept'))[['from_id_cat', 'to_id_concept']].
            rename(columns={'from_id_cat': 'from_id', 'to_id_concept': 'to_id'})
        )

        if self.test_mode and self.test_ids is not None:
            # Saving the category-concept rows of the test set
            self.test_category_concept = self.category_concept.loc[
                self.category_concept['to_id'].apply(lambda x: x in self.test_ids)
            ]
            self.test_cluster_concept = self.cluster_concept.loc[
                self.cluster_concept['to_id'].apply(lambda x: x in self.test_ids)
            ]
            # Removing the test set concepts from category-concept and cluster-concept tables
            self.category_concept = self.category_concept.loc[
                self.category_concept['to_id'].apply(lambda x: x not in self.test_ids)
            ]
            self.cluster_concept = self.cluster_concept.loc[
                self.cluster_concept['to_id'].apply(lambda x: x not in self.test_ids)
            ]

        category_concept_agg = self.category_concept.assign(
            id=self.category_concept.to_id.apply(lambda x: [x])
        )[['from_id', 'id']].groupby('from_id').agg(sum).reset_index()
        self.category_concept_dict = get_col_to_col_dict(category_concept_agg, 'from_id', 'id')

        cluster_concept_agg = self.cluster_concept.assign(
            id=self.cluster_concept.to_id.apply(lambda x: [x])
        )[['from_id', 'id']].groupby('from_id').agg(sum).reset_index()
        self.cluster_concept_dict = get_col_to_col_dict(cluster_concept_agg, 'from_id', 'id')

        category_cluster_agg = self.category_cluster.assign(
            id=self.category_cluster.to_id.apply(lambda x: [x])
        )[['from_id', 'id']].groupby('from_id').agg(sum).reset_index()
        self.category_cluster_dict = get_col_to_col_dict(category_cluster_agg, 'from_id', 'id')

    def load_anchor_page_dict(self):
        """
        Loads category to anchor page list dictionary using the direct category-anchor table and the
        category child-parent relations table.
        Returns:
            None
        """
        db_manager = DB(self.db_config)
        # Load the direct anchor page table
        base_anchors = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT from_id, to_id FROM graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage"
        ), ['from_id', 'to_id'])
        # Aggregate the direct anchors of each category into a list
        base_anchors['to_id'] = base_anchors['to_id'].apply(lambda x: [x])
        base_anchors = base_anchors.groupby('from_id').agg(sum).reset_index().rename(columns={
            'from_id': 'category_id', 'to_id': 'anchor_ids'
        })
        # Add the depth of the category to the dataframe
        base_anchors = pd.merge(base_anchors, self.ontology_categories,
                                on='category_id')[['category_id', 'depth', 'anchor_ids']]

        # Start the process of aggregating with children, bottom-up
        anchors = base_anchors.loc[base_anchors.depth == 4]
        for depth in range(3, -1, -1):
            # Get current categories
            current_cat_df = self.ontology_categories.loc[
                self.ontology_categories.depth == depth, ['category_id', 'depth']
            ]
            # Join each current category with its children in the anchors calculated so far
            current_relationships = pd.merge(current_cat_df, self.category_category, left_on='category_id',
                                             right_on='to_id', how='left').drop(columns=['to_id'])
            current_relationships = (pd.merge(current_relationships, anchors, left_on='from_id',
                                     right_on='category_id', how='left', suffixes=('', '_tgt')).
                                     drop(columns=['from_id']))
            # If the left join has yielded null values, turn them into empty lists
            current_relationships['anchor_ids'] = current_relationships['anchor_ids'].apply(
                lambda x: x if isinstance(x, list) else []
            )
            # Include the direct anchors of the current categories
            current_relationships = pd.concat(
                [current_relationships, base_anchors.loc[base_anchors.depth == depth]], axis=0
            )
            # Aggregate everything
            all_new_anchors = (current_relationships[['category_id', 'anchor_ids']].
                               groupby('category_id').agg(sum).reset_index())
            all_new_anchors['depth'] = depth
            anchors = pd.concat([anchors, all_new_anchors], axis=0)
        # Remove duplicates in each of the entries
        anchors['anchor_ids'] = anchors['anchor_ids'].apply(lambda x: list(set(x)))

        category_ids = anchors.category_id.values.tolist()
        anchor_lists = anchors.anchor_ids.values.tolist()
        depths_list = anchors.depth.values.tolist()
        self.category_anchors_dict = {category_ids[i]: {'anchors': anchor_lists[i], 'depth': depths_list[i]}
                                      for i in range(len(category_ids))}

    def compute_category_anchors_using_references(self):
        assert self.category_category is not None and self.ontology_categories is not None
        print('computing category anchors')
        anchors = self.ontology_categories.loc[self.ontology_categories.depth == 5, ['category_id', 'depth', 'id']].\
            copy()
        anchors['anchor_ids'] = anchors['id'].apply(lambda x: [x])
        for depth in range(4, -1, -1):
            current_cat_df = self.ontology_categories.loc[
                self.ontology_categories.depth == depth, ['category_id', 'depth', 'id']
            ]
            current_relationships = pd.merge(current_cat_df, self.category_category, left_on='category_id',
                                             right_on='to_id', how='left').drop(columns=['to_id'])
            current_relationships = (pd.merge(current_relationships, anchors, left_on='from_id',
                                     right_on='category_id', how='left', suffixes=('', '_tgt')).
                                     drop(columns=['from_id']))
            current_relationships['anchor_ids'] = current_relationships['anchor_ids'].apply(
                lambda x: x if isinstance(x, list) else []
            )
            new_anchors = (current_relationships[['category_id', 'anchor_ids']].
                           groupby('category_id').agg(sum).reset_index())
            base_anchors = current_cat_df.assign(anchor_ids=current_cat_df['id'].apply(lambda x: [x]))
            all_new_anchors = pd.merge(new_anchors, base_anchors, on='category_id', suffixes=('_children', '_base'))
            all_new_anchors['anchor_ids'] = all_new_anchors.apply(
                lambda x: x['anchor_ids_children'] + x['anchor_ids_base'], axis=1
            )
            all_new_anchors = all_new_anchors[['category_id', 'depth', 'id', 'anchor_ids']]
            anchors = pd.concat([anchors, all_new_anchors], axis=0)
        anchors = anchors.loc[anchors.depth < 5]
        category_ids = anchors.category_id.values.tolist()
        anchor_lists = anchors.anchor_ids.values.tolist()
        depths_list = anchors.depth.values.tolist()
        return {category_ids[i]: {'anchors': anchor_lists[i], 'depth': depths_list[i]}
                for i in range(len(category_ids))}

    def compute_symmetric_concept_concept_matrix(self):
        adj, row_dict, _ = (create_graph_from_df(
            self.concept_concept_graphscore, 'from_id', 'to_id', 'score',
            pool_rows_and_cols=True, make_symmetric=True)
        )
        self.symmetric_concept_concept_matrix['concept_id_to_index'] = row_dict
        self.symmetric_concept_concept_matrix['matrix'] = adj

    def compute_precalculated_similarity_matrices(self):
        # First, computations for depth 4 categories
        depth4_categories_list = sorted([x for x in self.category_anchors_dict.keys()
                                         if self.category_anchors_dict[x]['depth'] == 4])

        self.symmetric_concept_concept_matrix['d4_cat_index_to_id'] = dict(enumerate(depth4_categories_list))
        self.symmetric_concept_concept_matrix['d4_cat_id_to_index'] = (
            invert_dict(self.symmetric_concept_concept_matrix['d4_cat_index_to_id']))

        # Anchors
        self.symmetric_concept_concept_matrix['d4_cat_anchors'] = {
            x: [self.symmetric_concept_concept_matrix['concept_id_to_index'][y]
                for y in self.category_anchors_dict[
                    self.symmetric_concept_concept_matrix['d4_cat_index_to_id'][x]]['anchors']
                if y in self.symmetric_concept_concept_matrix['concept_id_to_index']]
            for x in range(len(depth4_categories_list))
        }
        self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'] = csr_matrix(
            ([len(v) for k, v in self.symmetric_concept_concept_matrix['d4_cat_anchors'].items()],
             ([0] * len(self.symmetric_concept_concept_matrix['d4_cat_anchors']),
              [k for k, v in self.symmetric_concept_concept_matrix['d4_cat_anchors'].items()])),
            shape=(1, len(self.symmetric_concept_concept_matrix['d4_cat_anchors']))
        )

        # Anchor-based matrices
        self.symmetric_concept_concept_matrix['matrix_concept_cat_anchors'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix'][
                        self.symmetric_concept_concept_matrix['d4_cat_anchors'][x], :].sum(axis=0)
                        )
             for x in range(len(depth4_categories_list))]
        ).transpose().tocsr()
        self.symmetric_concept_concept_matrix['matrix_cat_cat_anchors'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix_concept_cat_anchors'][
                        self.symmetric_concept_concept_matrix['d4_cat_anchors'][x], :].sum(axis=0)
                        )
             for x in range(len(depth4_categories_list))]
        )

        # Concepts
        self.symmetric_concept_concept_matrix['d4_cat_concepts'] = {
            x: [self.symmetric_concept_concept_matrix['concept_id_to_index'][y]
                for y in self.category_concept_dict.get(
                    self.symmetric_concept_concept_matrix['d4_cat_index_to_id'][x], [])
                if y in self.symmetric_concept_concept_matrix['concept_id_to_index']]
            for x in range(len(depth4_categories_list))
        }
        self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'] = csr_matrix(
            ([len(v) for k, v in self.symmetric_concept_concept_matrix['d4_cat_concepts'].items()],
             ([0] * len(self.symmetric_concept_concept_matrix['d4_cat_concepts']),
              [k for k, v in self.symmetric_concept_concept_matrix['d4_cat_concepts'].items()])),
            shape=(1, len(self.symmetric_concept_concept_matrix['d4_cat_concepts']))
        )

        # Concept-based matrices
        self.symmetric_concept_concept_matrix['matrix_concept_cat_concepts'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix'][
                        self.symmetric_concept_concept_matrix['d4_cat_concepts'][x], :].sum(axis=0)
                        )
             for x in range(len(depth4_categories_list))]
        ).transpose().tocsr()
        self.symmetric_concept_concept_matrix['matrix_cat_cat_concepts'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix_concept_cat_concepts'][
                        self.symmetric_concept_concept_matrix['d4_cat_concepts'][x], :].sum(axis=0)
                        )
             for x in range(len(depth4_categories_list))]
        )

        # A few computations for depth 3 categories
        depth3_categories_list = sorted([x for x in self.category_anchors_dict.keys()
                                         if self.category_anchors_dict[x]['depth'] == 3])
        self.symmetric_concept_concept_matrix['d3_cat_index_to_id'] = dict(enumerate(depth3_categories_list))
        self.symmetric_concept_concept_matrix['d3_cat_id_to_index'] = (
            invert_dict(self.symmetric_concept_concept_matrix['d3_cat_index_to_id']))
        self.symmetric_concept_concept_matrix['d3_to_d4'] = [
            [self.symmetric_concept_concept_matrix['d4_cat_id_to_index'][y]
             for y in self.category_category_dict[self.symmetric_concept_concept_matrix['d3_cat_index_to_id'][x]]]
            for x in range(len(depth3_categories_list))
        ]

        # Now, computations for clusters
        clusters_list = sorted(list(self.cluster_concept_dict.keys()))

        self.symmetric_concept_concept_matrix['cluster_index_to_id'] = dict(enumerate(clusters_list))
        self.symmetric_concept_concept_matrix['cluster_id_to_index'] = (
            invert_dict(self.symmetric_concept_concept_matrix['cluster_index_to_id']))

        # Concepts
        self.symmetric_concept_concept_matrix['cluster_concepts'] = {
            x: [self.symmetric_concept_concept_matrix['concept_id_to_index'][y]
                for y in self.cluster_concept_dict.get(
                    self.symmetric_concept_concept_matrix['cluster_index_to_id'][x], [])
                if y in self.symmetric_concept_concept_matrix['concept_id_to_index']]
            for x in range(len(clusters_list))
        }
        self.symmetric_concept_concept_matrix['cluster_concepts_lengths'] = csr_matrix(
            ([len(v) for k, v in self.symmetric_concept_concept_matrix['cluster_concepts'].items()],
             ([0] * len(self.symmetric_concept_concept_matrix['cluster_concepts']),
              [k for k, v in self.symmetric_concept_concept_matrix['cluster_concepts'].items()])),
            shape=(1, len(self.symmetric_concept_concept_matrix['cluster_concepts']))
        )

        # Concept-based matrices
        self.symmetric_concept_concept_matrix['matrix_concept_cluster_concepts'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix'][
                        self.symmetric_concept_concept_matrix['cluster_concepts'][x], :].sum(axis=0)
                        )
             for x in range(len(clusters_list))]
        ).transpose().tocsr()
        self.symmetric_concept_concept_matrix['matrix_cluster_cluster_concepts'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix_concept_cluster_concepts'][
                        self.symmetric_concept_concept_matrix['cluster_concepts'][x], :].sum(axis=0)
                        )
             for x in range(len(clusters_list))]
        )
        self.symmetric_concept_concept_matrix['matrix_cluster_cat_concepts'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix_concept_cat_concepts'][
                        self.symmetric_concept_concept_matrix['cluster_concepts'][x], :].sum(axis=0)
                        )
             for x in range(len(depth4_categories_list))]
        )
        self.symmetric_concept_concept_matrix['matrix_cluster_cat_anchors'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix_concept_cat_anchors'][
                        self.symmetric_concept_concept_matrix['cluster_concepts'][x], :].sum(axis=0)
                        )
             for x in range(len(depth4_categories_list))]
        )

    def get_concept_concept_similarity(self, concept_1_id, concept_2_id):
        self.load_data()
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if concept_1_id not in concepts or concept_2_id not in concepts:
            return None
        concept_1_index = concepts[concept_1_id]
        concept_2_index = concepts[concept_2_id]
        return self.symmetric_concept_concept_matrix['matrix'][concept_1_index, concept_2_index]

    def get_concept_cluster_similarity(self, concept_id, cluster_id, avg='linear'):
        self.load_data()
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        clusters = self.symmetric_concept_concept_matrix['cluster_id_to_index']
        if concept_id not in concepts or cluster_id not in clusters:
            return None
        concept_index = concepts[concept_id]
        cluster_index = clusters[cluster_id]
        score = self.symmetric_concept_concept_matrix['matrix_concept_cluster_concepts'][concept_index, cluster_index]
        denominator = self.symmetric_concept_concept_matrix['cluster_concepts_lengths'][0, cluster_index]
        return compute_average(score, denominator, avg)

    def get_cluster_cluster_similarity(self, cluster_1_id, cluster_2_id, avg='linear'):
        self.load_data()
        clusters = self.symmetric_concept_concept_matrix['cluster_id_to_index']
        if cluster_1_id not in clusters or cluster_2_id not in clusters:
            return None
        cluster_1_index = clusters[cluster_1_id]
        cluster_2_index = clusters[cluster_2_id]
        score = self.symmetric_concept_concept_matrix['matrix_cluster_cluster_concepts'][cluster_1_index, cluster_2_index]
        denominator = (
            self.symmetric_concept_concept_matrix['cluster_concepts_lengths'][0, cluster_1_index]
            * self.symmetric_concept_concept_matrix['cluster_concepts_lengths'][0, cluster_2_index]
        )
        return compute_average(score, denominator, avg)

    def get_concept_category_similarity(self, concept_id, category_id, avg='linear', coeffs=(1, 4)):
        self.load_data()
        d4_cats = self.symmetric_concept_concept_matrix['d4_cat_id_to_index']
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if category_id not in d4_cats or concept_id not in concepts:
            return None
        concept_index = concepts[concept_id]
        cat_index = d4_cats[category_id]
        s1 = self.symmetric_concept_concept_matrix['matrix_concept_cat_anchors'][concept_index, cat_index]
        s2 = self.symmetric_concept_concept_matrix['matrix_concept_cat_concepts'][concept_index, cat_index]
        l1 = self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'][0, cat_index]
        l2 = self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'][0, cat_index]
        return average_and_combine(s1, s2, l1, l2, avg, coeffs)

    def get_cluster_category_similarity(self, cluster_id, category_id, avg='linear', coeffs=(1, 4)):
        self.load_data()
        clusters = self.symmetric_concept_concept_matrix['cluster_id_to_index']
        d4_cats = self.symmetric_concept_concept_matrix['d4_cat_id_to_index']
        if cluster_id not in clusters or category_id not in d4_cats:
            return None
        cluster_index = clusters[cluster_id]
        category_index = d4_cats[category_id]
        s1 = self.symmetric_concept_concept_matrix['matrix_cluster_cat_concepts'][cluster_index, category_index]
        l1 = (
            self.symmetric_concept_concept_matrix['cluster_concepts_lengths'][0, cluster_index]
            * self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'][0, category_index]
        )
        s2 = self.symmetric_concept_concept_matrix['matrix_cluster_cat_anchors'][cluster_index, category_index]
        l2 = (
            self.symmetric_concept_concept_matrix['cluster_concepts_lengths'][0, cluster_index]
            * self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'][0, category_index]
        )
        return average_and_combine(s1, s2, l1, l2, avg, coeffs)

    def get_category_category_similarity(self, category_1_id, category_2_id, avg='linear', coeffs=(1, 4)):
        self.load_data()
        d4_cats = self.symmetric_concept_concept_matrix['d4_cat_id_to_index']
        if category_1_id not in d4_cats or category_2_id not in d4_cats:
            return None
        category_1_index = d4_cats[category_1_id]
        category_2_index = d4_cats[category_2_id]
        s1 = self.symmetric_concept_concept_matrix['matrix_cat_cat_anchors'][category_1_index, category_2_index]
        s2 = self.symmetric_concept_concept_matrix['matrix_cat_cat_concepts'][category_1_index, category_2_index]
        l1 = (
            self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'][0, category_1_index]
            * self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'][0, category_2_index]
        )
        l2 = (
            self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'][0, category_1_index]
            * self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'][0, category_2_index]
        )
        return average_and_combine(s1, s2, l1, l2, avg, coeffs)

    def get_concept_closest_concept(self, concept_id, top_n=1):
        self.load_data()
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if concept_id not in concepts:
            return None, None
        concept_index = concepts[concept_id]
        results = np.array(self.symmetric_concept_concept_matrix['matrix'][[concept_index], :].todense()).flatten()
        concepts_inverse = invert_dict(concepts)
        if top_n == 1:
            best_concept_index = np.argmax(results)
            best_concept = concepts_inverse[best_concept_index]
            best_score = results[best_concept_index]
            return [best_concept], [best_score]
        else:
            sorted_indices = np.argsort(results)[::-1]
            best_concept_indices = sorted_indices[:top_n]
            best_concepts = [concepts_inverse[i] for i in best_concept_indices]
            best_scores = [results[i] for i in best_concept_indices]
            return best_concepts, best_scores

    def get_concept_closest_cluster_of_category(self, concept_id, category_id, avg='linear', top_n=3):
        self.load_data()
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if concept_id not in concepts or category_id not in self.category_cluster_dict:
            return None
        clusters = self.symmetric_concept_concept_matrix['cluster_id_to_index']
        candidate_cluster_ids = self.category_cluster_dict[category_id]
        concept_index = concepts[concept_id]
        candidate_cluster_indices = [clusters[x] for x in candidate_cluster_ids]
        score = (self.symmetric_concept_concept_matrix['matrix_concept_cluster_concepts']
                 [[concept_index], candidate_cluster_indices])
        denominator = (self.symmetric_concept_concept_matrix['cluster_concepts_lengths']
                       [[0], candidate_cluster_indices])
        results = compute_average(score, denominator, avg)
        sorted_indices = np.argsort(results)[::-1]
        best_cluster_indices = sorted_indices[:top_n]
        best_clusters = [self.symmetric_concept_concept_matrix['cluster_index_to_id'][candidate_cluster_indices[i]]
                         for i in best_cluster_indices]
        best_scores = [results[i] for i in best_cluster_indices]
        return best_clusters, best_scores

    def get_concept_closest_category(self, concept_id, avg='log', coeffs=(1, 10), top_n=1,
                                     use_depth_3=False, return_clusters=None):
        self.load_data()
        d4_cat_indices = self.symmetric_concept_concept_matrix['d4_cat_index_to_id']
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if concept_id not in concepts:
            return None, None, None, None
        concept_index = concepts[concept_id]
        s1 = self.symmetric_concept_concept_matrix['matrix_concept_cat_anchors'][[concept_index], :]
        s2 = self.symmetric_concept_concept_matrix['matrix_concept_cat_concepts'][[concept_index], :]
        l1 = self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths']
        l2 = self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths']
        results = average_and_combine(s1, s2, l1, l2, avg, coeffs)
        if use_depth_3:
            results_d3 = np.array([sum(results[self.symmetric_concept_concept_matrix['d3_to_d4'][i]])
                                   for i in range(len(self.symmetric_concept_concept_matrix['d3_to_d4']))])
            results_d3 /= np.array([len(self.symmetric_concept_concept_matrix['d3_to_d4'][i])
                                    for i in range(len(self.symmetric_concept_concept_matrix['d3_to_d4']))])
            best_d3_index = np.argmax(results_d3)
            selected_d3_category = self.symmetric_concept_concept_matrix['d3_cat_index_to_id'][best_d3_index]
            result_indices = self.symmetric_concept_concept_matrix['d3_to_d4'][best_d3_index]
            new_results = results.copy()
            new_results[result_indices] += (np.max(new_results) - np.min(new_results)) + 0.1
        else:
            new_results = results
            selected_d3_category = None
        sorted_indices = np.argsort(new_results)[::-1]
        best_cat_indices = sorted_indices[:top_n]
        best_cats = [d4_cat_indices[i] for i in best_cat_indices]
        best_scores = [results[i] for i in best_cat_indices]
        if return_clusters is not None:
            best_clusters = [self.get_concept_closest_cluster_of_category(concept_id, cat, avg, top_n=return_clusters)
                             for cat in best_cats]
        else:
            best_clusters = None
        return best_cats, best_scores, selected_d3_category, best_clusters

    def get_ontology_concept_names_table(self, concepts_to_keep=None):
        self.load_data()
        results = self.ontology_concept_names
        if concepts_to_keep is not None:
            results = results.loc[results["id"].apply(lambda x: x in concepts_to_keep)]
        return results

    def get_ontology_category_names(self):
        self.load_data()
        return self.ontology_categories

    def get_ontology_category_info(self, cat_id):
        self.load_data()
        results = self.ontology_categories.loc[self.ontology_categories.category_id == cat_id].to_dict(orient='records')
        if len(results) > 0:
            return results[0]
        return None

    def get_non_ontology_concept_names(self):
        self.load_data()
        return self.non_ontology_concept_names

    def get_concept_concept_graphscore_table(self, concepts_to_keep=None):
        self.load_data()
        results = self.concept_concept_graphscore
        if concepts_to_keep is not None:
            results = results.loc[results.to_id.apply(lambda x: x in concepts_to_keep)]
        return results

    def get_category_to_category(self):
        self.load_data()
        return self.category_category

    def get_category_parent(self, child_id):
        self.load_data()
        results = self.category_category.loc[self.category_category.from_id == child_id, 'to_id'].values.tolist()
        if len(results) > 0:
            return results[0]
        return None

    def get_category_children(self, parent_id):
        self.load_data()
        results = self.category_category.loc[self.category_category.to_id == parent_id, 'from_id'].values.tolist()
        if len(results) > 0:
            return results
        return None

    def get_category_cluster_list(self, cat_id):
        self.load_data()
        return self.category_cluster_dict.get(cat_id, None)

    def get_category_concept_list(self, cat_id):
        self.load_data()
        return self.category_concept_dict.get(cat_id, None)

    def get_cluster_concept_list(self, cluster_id):
        self.load_data()
        return self.cluster_concept_dict.get(cluster_id, None)

    def get_category_concept_table(self, concepts_to_keep=None):
        self.load_data()
        results = self.category_concept
        if concepts_to_keep is not None:
            results = results.loc[results.to_id.apply(lambda x: x in concepts_to_keep)]
        return results

    def get_category_anchor_pages(self, category_id):
        self.load_data()
        return self.category_anchors_dict.get(category_id, [])

    def get_cluster_concepts(self, cluster_id):
        self.load_data()
        return self.cluster_concept_dict.get(cluster_id, [])

    def get_test_category_concept(self):
        self.load_data()
        return self.test_category_concept

    def get_test_cluster_concept(self):
        self.load_data()
        return self.test_cluster_concept
