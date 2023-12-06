from db_cache_manager.db import DB
import pandas as pd
from scipy.sparse import csr_array, csr_matrix

from graphai.core.common.common_utils import invert_dict
from graphai.core.interfaces.config_loader import load_db_config


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


class OntologyData:
    def __init__(self):
        self.loaded = False
        self.db_config = None
        self.ontology_concept_names = None
        self.ontology_categories = None
        self.non_ontology_concept_names = None
        self.concept_concept_graphscore = None
        self.ontology_and_anchor_concepts_id_to_index = None
        self.ontology_neighbor_concepts_id_to_index = None
        self.symmetric_concept_concept_matrix = dict()
        self.category_category = None
        self.category_concept = None
        self.category_concept_dict = None
        self.category_anchors_dict = None

    def load_data(self):
        if not self.loaded:
            self.db_config = load_db_config()
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
            # Finally computing aggregated anchor concept lists for each category
            self.compute_category_anchors()
            self.loaded = True

    def load_ontology_concept_names(self):
        db_manager = DB(self.db_config)
        self.ontology_concept_names = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT id, name FROM graph_ontology.Nodes_N_Concept WHERE is_ontology_concept=1"),
            ['id', 'name']
        )

    def load_ontology_categories(self):
        db_manager = DB(self.db_config)
        self.ontology_categories = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT a.id AS category_id, a.depth AS depth, b.id AS concept_id, b.name AS concept_name "
            "FROM graph_ontology.Nodes_N_Category a JOIN graph_ontology.Nodes_N_Concept b "
            "ON a.anchor_page_id=b.id "
            "WHERE b.is_ontology_category=1;"),
            ['category_id', 'depth', 'id', 'name']
        )

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

    def load_category_concept(self):
        db_manager = DB(self.db_config)
        self.category_concept = db_results_to_pandas_df(db_manager.execute_query(
            "SELECT from_id, to_id FROM graph_ontology.Edges_N_Category_N_Concept_T_ParentToChild"),
            ['from_id', 'to_id']
        )
        category_concept_agg = self.category_concept.assign(
            id=self.category_concept.to_id.apply(lambda x: [x])
        )[['from_id', 'id']].groupby('from_id').agg(sum).reset_index()
        category_ids = category_concept_agg['from_id'].values.tolist()
        concept_lists = category_concept_agg['id'].values.tolist()
        self.category_concept_dict = {category_ids[i]: concept_lists[i] for i in range(len(category_ids))}

    def compute_category_anchors(self):
        assert self.category_category is not None and self.ontology_categories is not None
        print('computing category anchors')
        anchors = self.ontology_categories.loc[self.ontology_categories.depth == 5, ['category_id', 'depth', 'id']].\
            copy()
        anchors['anchor_ids'] = anchors['id'].apply(lambda x: [x])
        for depth in range(4,-1,-1):
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
        self.category_anchors_dict = {category_ids[i]: anchor_lists[i] for i in range(len(category_ids))}

    def compute_symmetric_concept_concept_matrix(self):
        adj, row_dict, _ = (create_graph_from_df(
            self.concept_concept_graphscore, 'from_id', 'to_id', 'score',
            pool_rows_and_cols=True, make_symmetric=True)
        )
        self.symmetric_concept_concept_matrix['id_to_index'] = row_dict
        self.symmetric_concept_concept_matrix['matrix'] = adj

    def get_ontology_concept_names(self):
        self.load_data()
        return self.ontology_concept_names

    def get_ontology_category_names(self):
        self.load_data()
        return self.ontology_categories

    def get_non_ontology_concept_names(self):
        self.load_data()
        return self.non_ontology_concept_names

    def get_concept_concept_graphscore(self):
        self.load_data()
        return self.concept_concept_graphscore

    def get_category_to_category(self):
        self.load_data()
        return self.category_category

    def get_category_concept(self):
        self.load_data()
        return self.category_concept

    def get_category_anchor_pages(self):
        self.load_data()
        return self.category_anchors_dict
