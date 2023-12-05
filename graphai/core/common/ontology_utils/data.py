from db_cache_manager.db import DB
import pandas as pd
from graphai.core.interfaces.config_loader import load_db_config


def db_results_to_pandas_df(results, cols):
    return pd.DataFrame(results, columns=cols)


class OntologyData:
    def __init__(self):
        self.loaded = False
        self.ontology_concept_names = None
        self.concept_concept_graphscore = None
        self.category_concept = None

    def load_data(self):
        if not self.loaded:
            db_manager = DB(load_db_config())
            self.ontology_concept_names = db_results_to_pandas_df(db_manager.execute_query(
                "SELECT id, name FROM graph_ontology.Nodes_N_Concept WHERE is_ontology_concept=1"),
                ['id', 'name']
            )
            self.concept_concept_graphscore = db_results_to_pandas_df(db_manager.execute_query(
                "SELECT from_id, to_id, score FROM graph_ontology.Edges_N_Concept_N_Concept_T_Undirected"),
                ['from_id', 'to_id', 'score']
            )
            self.category_concept = db_results_to_pandas_df(db_manager.execute_query(
                "SELECT from_id, to_id FROM graph_ontology.Edges_N_Category_N_Concept_T_ParentToChild"),
                ['from_id', 'to_id']
            )
            self.loaded = True

    def get_ontology_concept_names(self):
        self.load_data()
        return self.ontology_concept_names

    def get_concept_concept_graphscore(self):
        self.load_data()
        return self.concept_concept_graphscore

    def get_category_concept(self):
        self.load_data()
        return self.category_concept
