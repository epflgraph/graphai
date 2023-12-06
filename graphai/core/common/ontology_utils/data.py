from db_cache_manager.db import DB
import pandas as pd
from graphai.core.interfaces.config_loader import load_db_config


def db_results_to_pandas_df(results, cols):
    return pd.DataFrame(results, columns=cols)


class OntologyData:
    def __init__(self):
        self.loaded = False
        self.db_config = None
        self.ontology_concept_names = None
        self.ontology_categories = None
        self.non_ontology_concept_names = None
        self.concept_concept_graphscore = None
        self.category_category = None
        self.category_concept = None
        self.category_anchors = None

    def load_data(self):
        if not self.loaded:
            self.db_config = load_db_config()
            self.load_ontology_concept_names()
            self.load_ontology_categories()
            self.load_non_ontology_concept_names()
            self.load_concept_concept_graphscore()
            self.load_category_category()
            self.load_category_concept()
            self.load_category_anchors()
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

    def load_category_anchors(self):
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
        self.category_anchors = anchors

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
        return self.category_anchors

