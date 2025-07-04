import numpy as np
from db_cache_manager.db import DB
import pandas as pd
import random
from scipy.sparse import csr_array, csr_matrix, vstack, spmatrix
from itertools import chain
from multiprocessing import Lock

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


def adjusted_exp(x, overlap=5.0):
    return (np.log(1.0 + overlap) / (np.exp(overlap / ((1 + overlap) * np.log(1 + overlap))))
            * np.exp(x / ((1 + overlap) * np.log(1 + overlap))))


def adjusted_exp_slope_1_point(overlap=5.0):
    return (1 + overlap) * np.log(1 + overlap) * (np.log(1 + overlap) + overlap / ((1 + overlap) * np.log(1 + overlap)))


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


def ensure_nonzero_denominator(v):
    return v if v > 0 else 1


def compute_average_of_df(df, avg):
    df['score'] = df.apply(
        lambda x:
        x['score'] / ensure_nonzero_denominator(x['len']) if avg == 'linear'
        else x['score'] / np.log(ensure_nonzero_denominator(x['len']) + 1) if avg == 'log'
        else x['score'],
        axis=1
    )
    return df


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


def embeddings_table_exists():
    query = ("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
             "WHERE TABLE_SCHEMA='graph_ontology' AND TABLE_NAME='Edges_N_Concept_N_Concept_T_Embeddings';")
    db_manager = DB(config['database'])
    result = db_manager.execute_query(query)[0][0]
    return result == 1


def execute_single_entity_concepts_and_anchors_query(concepts_query, anchors_query, entity_id):
    db_manager = DB(config['database'])
    results_concepts = db_manager.execute_query(concepts_query, values=(entity_id,))
    results_anchors = db_manager.execute_query(anchors_query, values=(entity_id,))
    return results_concepts, results_anchors


def execute_multi_entity_concepts_and_anchors_query(concepts_query, anchors_query, entity_ids):
    db_manager = DB(config['database'])
    results_concepts = db_manager.execute_query(concepts_query, values=tuple(entity_ids))
    results_anchors = db_manager.execute_query(anchors_query, values=tuple(entity_ids))
    return results_concepts, results_anchors


def combine_concept_and_anchor_scores(concepts_query, anchors_query, entity_id, avg, coeffs, top_n,
                                      d4_cat_id_to_index, concept_lengths, anchor_lengths):
    if isinstance(entity_id, list):
        results_concepts, results_anchors = execute_multi_entity_concepts_and_anchors_query(
            concepts_query, anchors_query, entity_id
        )
    else:
        results_concepts, results_anchors = execute_single_entity_concepts_and_anchors_query(
            concepts_query, anchors_query, entity_id
        )
    results_concepts = pd.DataFrame(results_concepts, columns=['category_id', 'score']).assign(coeff=coeffs[1])
    results_concepts['len'] = results_concepts['category_id'].apply(
        lambda x: concept_lengths[0, d4_cat_id_to_index[x]]
    )
    results_anchors = pd.DataFrame(results_anchors, columns=['category_id', 'score']).assign(coeff=coeffs[0])
    results_anchors['len'] = results_anchors['category_id'].apply(
        lambda x: anchor_lengths[0, d4_cat_id_to_index[x]]
    )
    results_combined = pd.concat([results_concepts, results_anchors], axis=0)
    results_combined = compute_average_of_df(results_combined, avg)
    results_combined['score'] = results_combined['score'] * results_combined['coeff'] / sum(coeffs)
    results_combined = results_combined[['category_id', 'score']].groupby('category_id'). \
        sum().reset_index().sort_values('score', ascending=False).head(top_n)
    return results_combined


class OntologyData:
    def __init__(self, test_mode=False, **kwargs):
        self.loaded = False
        self.db_config = None
        self.ontology_concept_names = None
        self.ontology_concept_to_name_dict = None
        self.ontology_categories = None
        self.category_depth_dict = None
        self.non_ontology_concept_names = None
        self.concept_concept_graphscore = None
        self.concept_edge_counts = None
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
        self.edge_count_threshold = kwargs.get('adaptive_threshold', 20)
        # Parameters for test mode
        self.test_mode = test_mode
        self.random_state = kwargs.get('random_state', 0)
        self.test_ratio = kwargs.get('test_ratio', 0.0)
        self.sampling_method = kwargs.get('sampling_method', 'weighted')
        self.weighted_min_n = kwargs.get('min_n', 0)
        assert not self.test_mode or self.sampling_method in ['simple', 'weighted', 'weighted_log']
        self.test_ids = None
        self.test_concept_names = None
        self.test_category_concept = None
        self.test_cluster_concept = None
        self.load_lock = Lock()

    def load_data(self):
        with self.load_lock:
            try:
                if not self.loaded:
                    self.db_config = config['database']
                    # First loading concepts: ontology concepts, anchor concepts, and ontology-neighborhood concepts
                    self.load_ontology_concept_names()
                    self.load_ontology_categories()
                    self.load_non_ontology_concept_names()
                    # Now loading category-category and category-concept tables
                    self.load_category_category()
                    self.load_category_concept()
                    # Now loading the concept-concept table and the matrix from neighborhood to ontology/anchors
                    self.load_concept_concept_graphscore()
                    self.compute_symmetric_concept_concept_matrix()
                    # Now loading aggregated anchor concept lists for each category
                    self.load_anchor_page_dict()
                    # Finally, we compute aggregated matrices that map each concept to each category
                    self.compute_precalculated_similarity_matrices()
                    self.loaded = True
                success = True
            except Exception as e:
                print(e)
                success = False
        return success

    def load_ontology_concept_names(self):
        db_manager = DB(self.db_config)
        self.ontology_concept_names = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT id, name FROM graph_ontology.Nodes_N_Concept WHERE is_ontology_concept=1"),
            ['id', 'name']
        )
        self.ontology_concept_names['id'] = self.ontology_concept_names['id'].astype(str)
        self.ontology_concept_to_name_dict = get_col_to_col_dict(self.ontology_concept_names, 'id', 'name')

    def load_ontology_categories(self):
        db_manager = DB(self.db_config)
        self.ontology_categories = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT a.id AS category_id, a.depth AS depth, b.id AS concept_id, b.name AS concept_name "
            "FROM graph_ontology.Nodes_N_Category a JOIN graph_ontology.Nodes_N_Concept b "
            "ON a.reference_page_id=b.id "
            "WHERE b.is_ontology_category=1;"),
            ['category_id', 'depth', 'id', 'name']
        )
        self.ontology_categories['id'] = self.ontology_categories['id'].astype(str)
        cat_ids = self.ontology_categories.category_id.values.tolist()
        depths = self.ontology_categories.depth.values.tolist()
        self.category_depth_dict = {cat_ids[i]: depths[i] for i in range(len(cat_ids))}

    def load_non_ontology_concept_names(self):
        db_manager = DB(self.db_config)
        self.non_ontology_concept_names = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT id, name FROM graph_ontology.Nodes_N_Concept WHERE is_ontology_neighbour=1"),
            ['id', 'name']
        )
        self.non_ontology_concept_names['id'] = self.non_ontology_concept_names['id'].astype(str)

    def load_concept_concept_graphscore(self):
        db_manager = DB(self.db_config)
        self.concept_concept_graphscore = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT from_id, to_id, score FROM graph_ontology.Edges_N_Concept_N_Concept_T_Undirected"),
            ['from_id', 'to_id', 'score']
        )
        self.concept_edge_counts = get_col_to_col_dict(
            pd.concat([
                self.concept_concept_graphscore,
                self.concept_concept_graphscore.rename(columns={'from_id': 'to_id', 'to_id': 'from_id'})
            ], axis=0).groupby('from_id').count().reset_index(), 'from_id', 'to_id'
        )
        print('Edge counts computed')

    def load_category_category(self):
        db_manager = DB(self.db_config)
        self.category_category = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT from_id, to_id FROM graph_ontology.Edges_N_Category_N_Category_T_ChildToParent;"),
            ['from_id', 'to_id']
        )
        category_category_agg = self.category_category.assign(
            id=self.category_category.from_id.apply(lambda x: [x])
        )[['to_id', 'id']].groupby('to_id').agg("sum").reset_index()
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

        if self.test_mode:
            test_n = int(self.ontology_concept_names.shape[0] * self.test_ratio)
            if test_n > 0:
                all_ids = self.ontology_concept_names['id'].values.tolist()
                random.seed(self.random_state)
                if self.sampling_method == 'simple':
                    self.test_ids = random.sample(all_ids, test_n)
                else:
                    weights = (self.category_concept.groupby('from_id').
                               count().reset_index().rename(columns={'to_id': 'count'}))
                    weights = pd.merge(self.category_concept, weights, on='from_id')
                    weight_dict = get_col_to_col_dict(weights, 'to_id', 'count')
                    weight_list = [weight_dict[i] for i in all_ids]
                    if self.weighted_min_n > 0:
                        indices_to_keep = np.argwhere(np.array(weight_list) > self.weighted_min_n).flatten().tolist()
                        all_ids = [all_ids[i] for i in indices_to_keep]
                        weight_list = [weight_list[i] for i in indices_to_keep]
                    if self.sampling_method == 'weighted':
                        weight_list = [1.0 / x for x in weight_list]
                    else:
                        weight_list = [1.0 / np.log(1 + x) for x in weight_list]
                    probabilities = np.array(weight_list) / sum(weight_list)
                    self.test_ids = np.random.choice(all_ids, test_n, replace=False, p=probabilities)
                self.test_concept_names = self.ontology_concept_names.loc[
                    self.ontology_concept_names['id'].apply(lambda x: x in self.test_ids)
                ]
                self.ontology_concept_names = self.ontology_concept_names.loc[
                    self.ontology_concept_names['id'].apply(lambda x: x not in self.test_ids)
                ]
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
        )[['from_id', 'id']].groupby('from_id').agg("sum").reset_index()
        self.category_concept_dict = get_col_to_col_dict(category_concept_agg, 'from_id', 'id')

        cluster_concept_agg = self.cluster_concept.assign(
            id=self.cluster_concept.to_id.apply(lambda x: [x])
        )[['from_id', 'id']].groupby('from_id').agg("sum").reset_index()
        self.cluster_concept_dict = get_col_to_col_dict(cluster_concept_agg, 'from_id', 'id')

        category_cluster_agg = self.category_cluster.assign(
            id=self.category_cluster.to_id.apply(lambda x: [x])
        )[['from_id', 'id']].groupby('from_id').agg("sum").reset_index()
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
        base_anchors = base_anchors.groupby('from_id').agg("sum").reset_index().rename(columns={
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
                               groupby('category_id').agg("sum").reset_index())
            all_new_anchors['depth'] = depth
            anchors = pd.concat([anchors, all_new_anchors], axis=0)
        # Remove duplicates in each of the entries
        anchors['anchor_ids'] = anchors['anchor_ids'].apply(lambda x: list(set(x)))

        category_ids = anchors.category_id.values.tolist()
        anchor_lists = anchors.anchor_ids.values.tolist()
        depths_list = anchors.depth.values.tolist()
        self.category_anchors_dict = {category_ids[i]: {'anchors': anchor_lists[i], 'depth': depths_list[i]}
                                      for i in range(len(category_ids))}

    def compute_symmetric_concept_concept_matrix(self):
        """
        Loads the concept-concept matrix and creates its index dictionary
        Returns:
            None
        """
        adj, row_dict, _ = (create_graph_from_df(
            self.concept_concept_graphscore, 'from_id', 'to_id', 'score',
            pool_rows_and_cols=True, make_symmetric=True)
        )
        self.symmetric_concept_concept_matrix['concept_id_to_index'] = row_dict
        self.symmetric_concept_concept_matrix['matrix'] = adj

    def compute_precalculated_similarity_matrices(self):
        """
        Precomputes the matrices and index dictionaries for similarities between concepts, clusters, and categories.
        Returns:
            None
        """
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
                        self.symmetric_concept_concept_matrix['cluster_concepts'].get(x, []), :].sum(axis=0)
                        )
             for x in range(len(clusters_list))]
        ).transpose().tocsr()
        self.symmetric_concept_concept_matrix['matrix_cluster_cluster_concepts'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix_concept_cluster_concepts'][
                        self.symmetric_concept_concept_matrix['cluster_concepts'].get(x, []), :].sum(axis=0)
                        )
             for x in range(len(clusters_list))]
        )
        self.symmetric_concept_concept_matrix['matrix_cluster_cat_concepts'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix_concept_cat_concepts'][
                        self.symmetric_concept_concept_matrix['cluster_concepts'].get(x, []), :].sum(axis=0)
                        )
             for x in range(len(clusters_list))]
        )
        self.symmetric_concept_concept_matrix['matrix_cluster_cat_anchors'] = vstack(
            [csr_matrix(self.symmetric_concept_concept_matrix['matrix_concept_cat_anchors'][
                        self.symmetric_concept_concept_matrix['cluster_concepts'].get(x, []), :].sum(axis=0)
                        )
             for x in range(len(clusters_list))]
        )

    def get_concept_concept_similarity(self, concept_1_id, concept_2_id):
        """
        Returns the similarity score between two concepts
        Args:
            concept_1_id: ID of concept 1
            concept_2_id: ID of concept 2

        Returns:
            Similarity score
        """
        load_success = self.load_data()
        if not load_success:
            return None
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if concept_1_id not in concepts or concept_2_id not in concepts:
            return None
        concept_1_index = concepts[concept_1_id]
        concept_2_index = concepts[concept_2_id]
        return self.symmetric_concept_concept_matrix['matrix'][concept_1_index, concept_2_index]

    def get_concept_cluster_similarity(self, concept_id, cluster_id, avg='linear'):
        """
        Returns the similarity score between a concept and a cluster
        Args:
            concept_id: ID of concept
            cluster_id: ID of cluster
            avg: Averaging method

        Returns:
            Similarity score
        """
        load_success = self.load_data()
        if not load_success:
            return None
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
        """
        Returns the similarity score between two clusters
        Args:
            cluster_1_id: ID of cluster 1
            cluster_2_id: ID of cluster 2
            avg: Averaging method

        Returns:
            Similarity score
        """
        load_success = self.load_data()
        if not load_success:
            return None
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
        """
        Returns the similarity score between a concept and a category
        Args:
            concept_id: ID of cluster
            category_id: ID of category
            avg: Averaging method
            coeffs: Coefficients for anchors and concepts

        Returns:
            Similarity score
        """
        load_success = self.load_data()
        if not load_success:
            return None
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
        """
        Returns the similarity score between a cluster and a category
        Args:
            cluster_id: ID of cluster
            category_id: ID of category
            avg: Averaging method
            coeffs: Coefficients for anchors and concepts

        Returns:
            Similarity score
        """
        load_success = self.load_data()
        if not load_success:
            return None
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
        """
        Returns the similarity score between two categories
        Args:
            category_1_id: ID of category 1
            category_2_id: ID of category 2
            avg: Averaging method
            coeffs: Coefficients for anchors and concepts

        Returns:
            Similarity score
        """
        load_success = self.load_data()
        if not load_success:
            return None
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
        """
        Finds the closest concept to a given concept
        Args:
            concept_id: Concept ID
            top_n: Number of top concepts to return

        Returns:
            Top concepts and their scores
        """
        load_success = self.load_data()
        if not load_success:
            return None, None
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

    def get_concept_closest_concept_embedding(self, concept_id, top_n=1):
        load_success = self.load_data()
        if not load_success:
            return None, None
        db_manager = DB(config['database'])
        query = """
        SELECT to_id, score FROM graph_ontology.Edges_N_Concept_N_Concept_T_Embeddings
        WHERE from_id=%s;
        """
        results = db_manager.execute_query(query, values=(concept_id, ))
        if len(results) == 0:
            return None, None
        results = pd.DataFrame(results, columns=['concept_id', 'score'])
        results = results.sort_values('score', ascending=False).head(top_n)
        return results.concept_id.values.tolist(), results.score.values.tolist()

    def _get_concept_closest_cluster_of_category(self, concept_id, category_id, avg='linear', top_n=3):
        """
        For a given concept and category, finds the top clusters under that category in terms of
        similarity to the concept.
        Args:
            concept_id: Concept ID
            category_id: Category ID
            avg: Averaging method
            top_n: Number of top clusters to return

        Returns:
            Top clusters and their similarity scores with the concept
        """
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if concept_id not in concepts or category_id not in self.category_cluster_dict:
            return None
        clusters = self.symmetric_concept_concept_matrix['cluster_id_to_index']
        candidate_cluster_ids = self.category_cluster_dict[category_id]
        concept_index = concepts[concept_id]
        # We do .get() because it is possible that there are some empty clusters, i.e. without a row in cluster-concept,
        # but with a row in category-cluster.
        candidate_cluster_indices = [clusters.get(x, None) for x in candidate_cluster_ids]
        candidate_cluster_indices = [x for x in candidate_cluster_indices if x is not None]
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

    def _go_through_depth_3(self, results):
        """
        Internal method, computes the closest depth-4 categories by going through depth-3 categories
        Args:
            results: Similarity score of the entity with each depth 4 category

        Returns:
            Modified depth-4 scores based on the best depth-3 category, plus the depth-3 category itself
        """
        results_d3 = np.array([sum(results[self.symmetric_concept_concept_matrix['d3_to_d4'][i]])
                               for i in range(len(self.symmetric_concept_concept_matrix['d3_to_d4']))])
        results_d3 /= np.array([len(self.symmetric_concept_concept_matrix['d3_to_d4'][i])
                                for i in range(len(self.symmetric_concept_concept_matrix['d3_to_d4']))])
        best_d3_index = np.argmax(results_d3)
        selected_d3_category = self.symmetric_concept_concept_matrix['d3_cat_index_to_id'][best_d3_index]
        result_indices = self.symmetric_concept_concept_matrix['d3_to_d4'][best_d3_index]
        new_results = results.copy()
        new_results[result_indices] += (np.max(new_results) - np.min(new_results)) + 0.1
        return new_results, selected_d3_category

    def _compute_closest_category_result(self, s1, s2, l1, l2, avg, coeffs, use_depth_3, top_n, d4_cat_indices,
                                         entity_edge_count=None, edge_count_threshold=None):
        """
        Internal method. Computes the closest category to an entity based on its anchor and concept score lists,
        as well as other parameters.
        Args:
            s1: Score list for anchors
            s2: Score list for concepts
            l1: Length list for anchors
            l2: Length list for concepts
            avg: Averaging method
            coeffs: Tuple of coefficients for the weighted average of anchor and concept scores, respectively
            use_depth_3: Whether to go through depth-3 categories or look at depth-4 directly
            top_n: Number of categories to return
            d4_cat_indices: Index dictionary for depth-4 categories

        Returns:
            Best categories, their scores, and the parent depth-3 category if use_depth_3==True
        """
        if avg == 'adaptive':
            if entity_edge_count is not None:
                if edge_count_threshold is None:
                    edge_count_threshold = self.edge_count_threshold
                if entity_edge_count > edge_count_threshold:
                    avg = 'linear'
                else:
                    avg = 'log'
            else:
                avg = 'log'
        results = average_and_combine(s1, s2, l1, l2, avg, coeffs)
        if use_depth_3:
            new_results, selected_d3_category = self._go_through_depth_3(results)
        else:
            new_results, selected_d3_category = results, None
        sorted_indices = np.argsort(new_results)[::-1]
        best_cat_indices = sorted_indices[:top_n]
        best_cats = [d4_cat_indices[i] for i in best_cat_indices]
        best_scores = [results[i] for i in best_cat_indices]
        return best_cats, best_scores, selected_d3_category, avg

    def get_concept_closest_category(self, concept_id, avg='log', coeffs=(1, 10), top_n=1,
                                     use_depth_3=False, return_clusters=None, adaptive_threshold=None):
        """
        Finds the closest category to a given concept
        Args:
            concept_id: Concept ID
            avg: Averaging method. Options are ('linear', 'log', and 'none')
            coeffs: Coefficients for averaging of the scores anchors and concepts
            top_n: Number of top categories to return
            use_depth_3: Whether to go through depth-3 or directly use depth-4
            return_clusters: Number of top clusters to return for each top category. If None, clusters are
                not returned.

        Returns:
            Top categories, their scores, parent depth-3 category if use_depth_3==True, and top clusters of
            each top category if return_clusters is not None.
        """
        load_success = self.load_data()
        if not load_success:
            return None, None, None, None
        d4_cat_indices = self.symmetric_concept_concept_matrix['d4_cat_index_to_id']
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if concept_id not in concepts:
            return None, None, None, None
        concept_index = concepts[concept_id]
        s1 = self.symmetric_concept_concept_matrix['matrix_concept_cat_anchors'][[concept_index], :]
        s2 = self.symmetric_concept_concept_matrix['matrix_concept_cat_concepts'][[concept_index], :]
        l1 = self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths']
        l2 = self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths']
        edge_count = self.concept_edge_counts[concept_id]
        best_cats, best_scores, selected_d3_category, avg = self._compute_closest_category_result(
            s1, s2, l1, l2, avg, coeffs, use_depth_3, top_n, d4_cat_indices, edge_count,
            edge_count_threshold=adaptive_threshold
        )
        if return_clusters is not None:
            best_clusters = [self._get_concept_closest_cluster_of_category(concept_id, cat, avg, top_n=return_clusters)
                             for cat in best_cats]
        else:
            best_clusters = None
        return best_cats, best_scores, selected_d3_category, best_clusters

    def get_concept_category_closest_embedding(self, concept_id, avg='log', coeffs=(1, 10),
                                               top_n=5, return_clusters=None):
        load_success = self.load_data()
        if not load_success:
            return None, None, None, None
        # If the embedding-based concept-concept similarity table does not exist, this method cannot run
        if not embeddings_table_exists():
            return None, None, None, None
        d4_cat_id_to_index = self.symmetric_concept_concept_matrix['d4_cat_id_to_index']
        anchor_lengths = self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths']
        concept_lengths = self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths']
        # Query to get closest category based on category concepts
        concepts_query = """
        SELECT c.from_id, SUM(a.score) as score_total FROM
        graph_ontology.Edges_N_Concept_N_Concept_T_Embeddings a
        INNER JOIN graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild b
        INNER JOIN graph_ontology.Edges_N_Category_N_ConceptsCluster_T_ParentToChild c
        ON a.to_id=b.to_id AND b.from_id=c.to_id WHERE a.from_id=%s
        GROUP BY c.from_id;
        """
        # Query to get closest category based on category anchor pages
        anchors_query = """
        SELECT b.from_id, SUM(a.score) as score_total FROM
        graph_ontology.Edges_N_Concept_N_Concept_T_Embeddings a
        INNER JOIN graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage b
        INNER JOIN graph_ontology.Nodes_N_Category c
        ON a.to_id=b.to_id AND b.from_id=c.id WHERE a.from_id=%s AND c.depth=4
        GROUP BY b.from_id;
        """
        results_combined = combine_concept_and_anchor_scores(concepts_query, anchors_query,
                                                             concept_id, avg, coeffs, top_n,
                                                             d4_cat_id_to_index, concept_lengths, anchor_lengths)
        if results_combined.shape[0] == 0:
            return None, None, None, None
        best_cats = results_combined.category_id.values.tolist()
        scores = results_combined.score.values.tolist()
        if return_clusters is not None:
            best_clusters = [self._get_concept_closest_cluster_of_category_embedding(concept_id,
                                                                                     cat,
                                                                                     top_n=return_clusters)
                             for cat in best_cats
                             ]
        else:
            best_clusters = None
        return best_cats, scores, None, best_clusters

    def _get_concept_closest_cluster_of_category_embedding(self, concept_id, cat, avg='log', top_n=None):
        cluster_id_to_index = self.symmetric_concept_concept_matrix['cluster_id_to_index']
        cluster_lengths = self.symmetric_concept_concept_matrix['cluster_concepts_lengths']
        db_manager = DB(config['database'])
        query = """
        SELECT b.from_id, SUM(a.score) as score_total FROM
        graph_ontology.Edges_N_Concept_N_Concept_T_Embeddings a
        INNER JOIN graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild b
        INNER JOIN graph_ontology.Edges_N_Category_N_ConceptsCluster_T_ParentToChild c
        ON a.to_id=b.to_id AND b.from_id=c.to_id WHERE a.from_id=%s AND c.from_id=%s
        GROUP BY b.from_id
        ORDER BY score_total DESC;
        """
        results = db_manager.execute_query(query, values=(concept_id, cat))
        results = pd.DataFrame(results, columns=['cluster_id', 'score'])
        results['len'] = results['cluster_id'].apply(lambda x: cluster_lengths[0, cluster_id_to_index[x]])
        results = compute_average_of_df(results, avg)
        results = (results[['cluster_id', 'score']].groupby('cluster_id').
                   sum().reset_index().sort_values('score', ascending=False)).head(top_n)
        best_clusters = results.cluster_id.values.tolist()
        scores = results.score.values.tolist()
        return best_clusters, scores

    def get_cluster_closest_category(self, cluster_id, avg='log', coeffs=(1, 10), top_n=1,
                                     use_depth_3=False):
        """
        Finds the closest category to a given cluster
        Args:
            cluster_id: Cluster ID
            avg: Averaging method. Options are ('linear', 'log', and 'none')
            coeffs: Coefficients for averaging of the scores anchors and concepts
            top_n: Number of top categories to return
            use_depth_3: Whether to go through depth-3 or directly use depth-4

        Returns:
            Top categories, their scores, and parent depth-3 category if use_depth_3==True.
        """
        load_success = self.load_data()
        if not load_success:
            return None, None, None
        d4_cat_indices = self.symmetric_concept_concept_matrix['d4_cat_index_to_id']
        clusters = self.symmetric_concept_concept_matrix['cluster_id_to_index']
        if cluster_id not in clusters:
            return None, None, None
        cluster_index = clusters[cluster_id]
        s1 = self.symmetric_concept_concept_matrix['matrix_cluster_cat_anchors'][[cluster_index], :]
        s2 = self.symmetric_concept_concept_matrix['matrix_cluster_cat_concepts'][[cluster_index], :]
        l1 = (self.symmetric_concept_concept_matrix['cluster_concepts_lengths'][0, cluster_index]
              * self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'])
        l2 = (self.symmetric_concept_concept_matrix['cluster_concepts_lengths'][0, cluster_index]
              * self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'])
        best_cats, best_scores, selected_d3_category, avg = self._compute_closest_category_result(
            s1, s2, l1, l2, avg, coeffs, use_depth_3, top_n, d4_cat_indices
        )
        return best_cats, best_scores, selected_d3_category

    def get_custom_cluster_closest_category(self, concept_ids, avg='log', coeffs=(1, 10), top_n=1,
                                            use_depth_3=False):
        """
        Finds the closest category to a custom cluster, provided as a list of concepts
        Args:
            concept_ids: List of concept IDs
            avg: Averaging method. Options are ('linear', 'log', and 'none')
            coeffs: Coefficients for averaging of the scores anchors and concepts
            top_n: Number of top categories to return
            use_depth_3: Whether to go through depth-3 or directly use depth-4

        Returns:
            Top categories, their scores, and parent depth-3 category if use_depth_3==True.
        """
        load_success = self.load_data()
        if not load_success:
            return None, None, None
        d4_cat_indices = self.symmetric_concept_concept_matrix['d4_cat_index_to_id']
        concepts = self.symmetric_concept_concept_matrix['concept_id_to_index']
        if any(concept_id not in concepts for concept_id in concept_ids):
            return None, None, None
        concept_indices = [concepts[concept_id] for concept_id in concept_ids]
        s1 = self.symmetric_concept_concept_matrix['matrix_concept_cat_anchors'][concept_indices, :].sum(axis=0)
        s2 = self.symmetric_concept_concept_matrix['matrix_concept_cat_concepts'][concept_indices, :].sum(axis=0)
        l1 = (len(concept_indices)
              * self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'])
        l2 = (len(concept_indices)
              * self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'])
        best_cats, best_scores, selected_d3_category, avg = self._compute_closest_category_result(
            s1, s2, l1, l2, avg, coeffs, use_depth_3, top_n, d4_cat_indices
        )
        return best_cats, best_scores, selected_d3_category

    def get_cluster_closest_category_embedding(self, cluster_id, avg='log', coeffs=(1, 10), top_n=1):
        """
        Finds the closest category to a custom cluster, provided as a list of concepts
        Args:
            concept_ids: List of concept IDs
            avg: Averaging method. Options are ('linear', 'log', and 'none')
            coeffs: Coefficients for averaging of the scores anchors and concepts
            top_n: Number of top categories to return
            use_depth_3: Whether to go through depth-3 or directly use depth-4

        Returns:
            Top categories, their scores, and parent depth-3 category if use_depth_3==True.
        """
        load_success = self.load_data()
        if not load_success:
            return None, None, None
        d4_cat_id_to_index = self.symmetric_concept_concept_matrix['d4_cat_id_to_index']
        anchor_lengths = self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths']
        concept_lengths = self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths']
        concepts_query = """
        SELECT c.from_id, SUM(a.score) as score_total FROM
        graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild cc
        INNER JOIN graph_ontology.Edges_N_Concept_N_Concept_T_Embeddings a
        INNER JOIN graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild b
        INNER JOIN graph_ontology.Edges_N_Category_N_ConceptsCluster_T_ParentToChild c
        ON cc.to_id=a.from_id AND a.to_id=b.to_id AND b.from_id=c.to_id WHERE cc.from_id=%s
        GROUP BY c.from_id;
        """
        # Query to get closest category based on category anchor pages
        anchors_query = """
        SELECT b.from_id, SUM(a.score) as score_total FROM
        graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild cc
        INNER JOIN graph_ontology.Edges_N_Concept_N_Concept_T_Embeddings a
        INNER JOIN graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage b
        INNER JOIN graph_ontology.Nodes_N_Category c
        ON cc.to_id=a.from_id AND a.to_id=b.to_id AND b.from_id=c.id WHERE cc.from_id=%s AND c.depth=4
        GROUP BY b.from_id;
        """
        results_combined = combine_concept_and_anchor_scores(concepts_query, anchors_query,
                                                             cluster_id, avg, coeffs, top_n,
                                                             d4_cat_id_to_index, concept_lengths, anchor_lengths)
        best_cats = results_combined.category_id.values.tolist()
        best_scores = results_combined.score.values.tolist()
        return best_cats, best_scores, None

    def get_custom_cluster_closest_category_embedding(self, concept_ids, avg='log', coeffs=(1, 10), top_n=1):
        """
        Finds the closest category to a custom cluster, provided as a list of concepts
        Args:
            concept_ids: List of concept IDs
            avg: Averaging method. Options are ('linear', 'log', and 'none')
            coeffs: Coefficients for averaging of the scores anchors and concepts
            top_n: Number of top categories to return
            use_depth_3: Whether to go through depth-3 or directly use depth-4

        Returns:
            Top categories, their scores, and parent depth-3 category if use_depth_3==True.
        """
        load_success = self.load_data()
        if not load_success:
            return None, None, None
        d4_cat_id_to_index = self.symmetric_concept_concept_matrix['d4_cat_id_to_index']
        anchor_lengths = self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths']
        concept_lengths = self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths']
        concepts_query = f"""
        SELECT c.from_id, SUM(a.score) as score_total FROM
        graph_ontology.Edges_N_Concept_N_Concept_T_Embeddings a
        INNER JOIN graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild b
        INNER JOIN graph_ontology.Edges_N_Category_N_ConceptsCluster_T_ParentToChild c
        ON a.to_id=b.to_id AND b.from_id=c.to_id
        WHERE a.from_id IN ({','.join(["%s"] * len(concept_ids))})
        GROUP BY c.from_id;
        """
        # Query to get closest category based on category anchor pages
        anchors_query = f"""
        SELECT b.from_id, SUM(a.score) as score_total FROM
        graph_ontology.Edges_N_Concept_N_Concept_T_Embeddings a
        INNER JOIN graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage b
        INNER JOIN graph_ontology.Nodes_N_Category c
        ON a.to_id=b.to_id AND b.from_id=c.id
        WHERE a.from_id IN ({','.join(["%s"] * len(concept_ids))}) AND c.depth=4
        GROUP BY b.from_id;
        """
        results_combined = combine_concept_and_anchor_scores(concepts_query, anchors_query,
                                                             concept_ids, avg, coeffs, top_n,
                                                             d4_cat_id_to_index, concept_lengths, anchor_lengths)
        best_cats = results_combined.category_id.values.tolist()
        best_scores = results_combined.score.values.tolist()
        return best_cats, best_scores, None

    def get_category_closest_category(self, category_id, avg='log', coeffs=(1, 10), top_n=1,
                                      use_depth_3=False):
        """
        Finds the closest category to a given category. As with the category-category similarity method, the similarity
        is composed of between-anchor and between-concept similarity, and there is no anchor-concept crossover.
        Args:
            category_id: Category ID
            avg: Averaging method. Options are ('linear', 'log', and 'none')
            coeffs: Coefficients for averaging of the scores anchors and concepts
            top_n: Number of top categories to return
            use_depth_3: Whether to go through depth-3 or directly use depth-4

        Returns:
            Top categories, their scores, and parent depth-3 category if use_depth_3==True.
        """
        load_success = self.load_data()
        if not load_success:
            return None, None, None
        d4_cat_indices = self.symmetric_concept_concept_matrix['d4_cat_index_to_id']
        if category_id not in d4_cat_indices:
            return None, None, None
        category_index = d4_cat_indices[category_id]
        s1 = self.symmetric_concept_concept_matrix['matrix_cat_cat_anchors'][[category_index], :]
        s2 = self.symmetric_concept_concept_matrix['matrix_cat_cat_concepts'][[category_index], :]
        l1 = (self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'][0, category_index]
              * self.symmetric_concept_concept_matrix['d4_cat_anchors_lengths'])
        l2 = (self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'][0, category_index]
              * self.symmetric_concept_concept_matrix['d4_cat_concepts_lengths'])
        best_cats, best_scores, selected_d3_category, avg = self._compute_closest_category_result(
            s1, s2, l1, l2, avg, coeffs, use_depth_3, top_n, d4_cat_indices
        )
        return best_cats, best_scores, selected_d3_category

    def get_ontology_concept_names_table(self, concepts_to_keep=None):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.ontology_concept_names
        if concepts_to_keep is not None:
            results = results.loc[results["id"].apply(lambda x: x in concepts_to_keep)]
        return results

    def get_ontology_category_names(self):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.ontology_categories

    def get_ontology_category_info(self, cat_id):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.ontology_categories.loc[self.ontology_categories.category_id == cat_id].to_dict(orient='records')
        if len(results) > 0:
            return results[0]
        return None

    def get_non_ontology_concept_names(self):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.non_ontology_concept_names

    def get_concept_concept_graphscore_table(self, concepts_to_keep=None):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.concept_concept_graphscore
        if concepts_to_keep is not None:
            results = results.loc[results.to_id.apply(lambda x: x in concepts_to_keep)]
        return results

    def get_category_to_category(self):
        load_success = self.load_data()
        if not load_success:
            return None
        return (self.category_category.
                rename(columns={'from_id': 'child_id', 'to_id': 'parent_id'}).to_dict(orient='records'))

    def get_category_parent(self, child_id):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.category_category.loc[self.category_category.from_id == child_id, 'to_id'].values.tolist()
        if len(results) > 0:
            return results[0]
        return None

    def get_category_branch(self, category_id):
        load_success = self.load_data()
        if not load_success:
            return None
        if category_id is None or category_id not in self.category_depth_dict:
            return None
        current_depth = self.category_depth_dict[category_id]
        current_category = category_id
        result_list = [category_id]
        while current_depth > 0:
            current_category = self.get_category_parent(current_category)
            result_list.append(current_category)
            current_depth -= 1
        return result_list

    def get_category_children(self, parent_id):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.category_category.loc[self.category_category.to_id == parent_id, 'from_id'].values.tolist()
        if len(results) > 0:
            return results
        return None

    def get_cluster_parent(self, cluster_id):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.category_cluster.loc[self.category_cluster.to_id == cluster_id, 'from_id'].values.tolist()
        if len(results) > 0:
            return results[0]
        return None

    def get_cluster_children(self, cluster_id):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.cluster_concept.loc[self.cluster_concept.from_id == cluster_id, 'to_id'].values.tolist()
        if len(results) > 0:
            return results
        return None

    def get_concept_parent_category(self, concept_id):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.category_concept.loc[self.category_concept.to_id == concept_id, 'from_id'].values.tolist()
        if len(results) > 0:
            return results[0]
        return None

    def get_concept_parent_cluster(self, concept_id):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.cluster_concept.loc[self.cluster_concept.to_id == concept_id, 'from_id'].values.tolist()
        if len(results) > 0:
            return results[0]
        return None

    def get_category_cluster_list(self, cat_id):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.category_cluster_dict.get(cat_id, None)

    def get_category_concept_list(self, cat_id):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.category_concept_dict.get(cat_id, None)

    def get_cluster_concept_list(self, cluster_id):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.cluster_concept_dict.get(cluster_id, None)

    def get_category_concept_table(self, concepts_to_keep=None):
        load_success = self.load_data()
        if not load_success:
            return None
        results = self.category_concept
        if concepts_to_keep is not None:
            results = results.loc[results.to_id.apply(lambda x: x in concepts_to_keep)]
        return results

    def get_category_cluster_table(self):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.category_cluster

    def get_category_anchor_pages(self, category_id):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.category_anchors_dict.get(category_id, [])

    def get_cluster_concepts(self, cluster_id):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.cluster_concept_dict.get(cluster_id, [])

    def get_concept_name(self, concept_id):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.ontology_concept_to_name_dict.get(concept_id, None)

    def get_concept_names_list(self, concept_ids):
        load_success = self.load_data()
        if not load_success:
            return None
        return [
            {'id': concept_id, 'name': self.ontology_concept_to_name_dict.get(concept_id, None)}
            for concept_id in concept_ids
        ]

    def get_test_concept_names(self):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.test_concept_names

    def get_test_category_concept(self):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.test_category_concept

    def get_test_cluster_concept(self):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.test_cluster_concept

    def get_root_category(self):
        load_success = self.load_data()
        if not load_success:
            return None
        return self.ontology_categories.loc[self.ontology_categories.depth == 0, 'category_id'].values.tolist()[0]

    def _generate_tree_structure(self, start=None):
        if start is None:
            start = self.get_root_category()
        children = self.get_category_children(start)
        if children is None:
            children = list()
        return [{
            'name': start,
            'id': start,
            'children': list(chain.from_iterable([self._generate_tree_structure(x) for x in children]))
        }]

    def generate_tree_structure(self, start=None):
        load_success = self.load_data()
        if not load_success:
            return None
        return self._generate_tree_structure(start)

    def generate_category_concept_dict(self):
        load_success = self.load_data()
        if not load_success:
            return None
        results = {}
        for category_id in self.category_cluster_dict:
            clusters = self.get_category_cluster_list(category_id)
            if clusters is None:
                continue
            current_results = list()
            for cluster_id in clusters:
                concepts = self.get_cluster_concept_list(cluster_id)
                if concepts is None:
                    continue
                concepts = [self.get_concept_name(concept) for concept in concepts]
                current_results.append({'cluster_id': cluster_id, 'concepts': concepts})
            results[category_id] = current_results
        return results
