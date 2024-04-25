from graphai.core.interfaces.config import config
from db_cache_manager.db import DB
from argparse import ArgumentParser
import pandas as pd


def check_if_concept_has_been_added(db_manager, concept_id):
    query = ("SELECT COUNT(*) FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild "
             "WHERE to_id=%s")
    result = db_manager.execute_query(query, values=(concept_id, ))[0][0]
    if result == 0:
        print(f'Concept {concept_id} not in cluster edge table!')
    elif result > 1:
        print(f'Concept {concept_id} duplicated!')
    else:
        return


def get_number_of_ontology_concepts(db_manager):
    query = "SELECT COUNT(*) FROM graph_ontology.Nodes_N_Concept WHERE is_ontology_concept=1;"
    result = db_manager.execute_query(query)[0][0]
    return result


def verify_category_cluster_relationship(db_manager, cluster_id, category_id):
    if cluster_id == '-':
        return True
    query = ("SELECT COUNT(*) FROM graph_ontology.Edges_N_Category_N_ConceptsCluster_T_ParentToChild "
             "WHERE from_id=%s AND to_id=%s;")
    result = db_manager.execute_query(query, values=(category_id, cluster_id))[0][0]
    if result == 0:
        return False
    return True


def add_concept_to_existing_cluster(db_manager, concept_id, cluster_id):
    # Insert concept into cluster-concept table
    query = ("INSERT INTO `graph_ontology`.`Edges_N_ConceptsCluster_N_Concept_T_ParentToChild` (`from_id`, `to_id`) "
             "VALUES (%s, %s)")
    db_manager.execute_query(query, values=(cluster_id, concept_id))

    # Fix the flags for the concept node
    query = ("UPDATE `graph_ontology`.`Nodes_N_Concept` SET "
             "`is_ontology_concept` = 1, "
             "`is_ontology_neighbour` = 0, "
             "`is_unused` = 0 "
             "WHERE id=%s;")
    db_manager.execute_query(query, values=(concept_id, ))


def add_concept_to_new_cluster(db_manager, concept_id, category_id):
    # Generate new cluster id
    query = ("SELECT CAST(MAX(CAST(from_id AS UNSIGNED)) + 1 AS CHAR(255)) "
             "FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild")
    new_cluster_id = db_manager.execute_query(query)[0][0]
    # Insert concept into cluster-concept table
    query = ("INSERT INTO `graph_ontology`.`Edges_N_ConceptsCluster_N_Concept_T_ParentToChild` (`from_id`, `to_id`) "
             "VALUES (%s, %s)")
    db_manager.execute_query(query, values=(new_cluster_id, concept_id))
    # Insert new cluster into category-cluster table
    query = ("INSERT INTO `graph_ontology`.`Edges_N_Category_N_ConceptsCluster_T_ParentToChild` (`from_id`, `to_id`) "
             "VALUES (%s, %s);")
    db_manager.execute_query(query, values=(category_id, new_cluster_id))
    # Fix the flags for the concept node
    query = ("UPDATE `graph_ontology`.`Nodes_N_Concept` SET "
             "`is_ontology_concept` = 1, "
             "`is_ontology_neighbour` = 0, "
             "`is_unused` = 0 "
             "WHERE id=%s;")
    db_manager.execute_query(query, values=(concept_id, ))


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--check_mode', action='store_true')
    args = parser.parse_args()
    input_csv = args.input_file
    check_mode = args.check_mode

    db_manager = DB(config['database'])
    input_df = pd.read_csv(input_csv, index_col='id')

    if check_mode:
        processed_concepts = input_df.loc[
            input_df['processed'] == 1, 'concept_id'
        ].values.tolist()
        processed_concepts = [str(x) for x in processed_concepts]
        for concept_id in processed_concepts:
            check_if_concept_has_been_added(db_manager, concept_id)
    else:
        df_to_process = input_df.copy()
        if 'chosen_category' in df_to_process.columns and 'candidate_categories' in df_to_process.columns:
            df_to_process = df_to_process.loc[
                (~pd.isna(df_to_process['chosen_category']))
                | (~pd.isna(df_to_process['candidate_categories']))
            ]
        elif 'chosen_category' in df_to_process.columns:
            df_to_process = df_to_process.loc[
                (~pd.isna(df_to_process['chosen_category']))
            ]

        df_to_process = df_to_process.loc[
            df_to_process['processed'] != 1
        ]
        for col in df_to_process.columns:
            if col == 'processed':
                continue
            df_to_process[col] = df_to_process[col].astype(str)

        concepts_dict = df_to_process[['concept_id', 'chosen_category', 'chosen_cluster']].to_dict(orient='records')
        print(df_to_process.shape)
        if df_to_process.shape[0] == 0:
            return
        starting_concept_count = get_number_of_ontology_concepts(db_manager)
        print('STARTING...')
        processed_concept_id_set = set()
        for concept_element in concepts_dict:
            if concept_element['chosen_category'] == '-':
                continue
            is_cluster_valid = verify_category_cluster_relationship(db_manager,
                                                                    concept_element['chosen_cluster'],
                                                                    concept_element['chosen_category'])
            if not is_cluster_valid:
                print(f'INVALID CLUSTER-CATEGORY relationship for {concept_element}')
                concept_element['chosen_cluster'] = '-'

            if concept_element['chosen_cluster'] == '-':
                add_concept_to_new_cluster(db_manager,
                                           concept_element['concept_id'],
                                           concept_element['chosen_category'])
            else:
                add_concept_to_existing_cluster(db_manager,
                                                concept_element['concept_id'],
                                                concept_element['chosen_cluster'])
            processed_concept_id_set.add(concept_element['concept_id'])
        final_concept_count = get_number_of_ontology_concepts(db_manager)
        print(f'EXPECTED TOTAL # OF CONCEPTS PROCESSED: {len(concepts_dict)}')
        print(f'TOTAL # OF CONCEPTS PROCESSED: {final_concept_count - starting_concept_count}')
        print('Writing modified dataframe to disk...')
        input_df['processed'] = input_df.apply(
            lambda x: 1 if str(x['concept_id']) in processed_concept_id_set else x['processed'],
            axis=1
        )
        input_df.to_csv(input_csv)


if __name__ == '__main__':
    main()
