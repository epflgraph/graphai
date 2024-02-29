-- When new concepts are going to be added to a new cluster, we need to calculate the next cluster id,
-- insert the concepts into the cluster-concept table, insert the new cluster into the category-cluster table,
-- and finally set the flags for the new concepts in the concepts table.

-- Setting the name of the new category and the list of concepts

SET @new_category_id := '<CATEGORY ID>';
SET @concepts_to_add := '<CONCEPT ID 1>,<CONCEPT ID 2>,...';

-- Computing the id of the new cluster

SET @next_cluster_number := (SELECT CAST(MAX(CAST(from_id AS UNSIGNED)) + 1 AS CHAR(255))
FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild
);

-- Adding the concepts to the cluster-concept table

SET @values_to_insert = REPLACE(@concepts_to_add, ',', CONCAT(', ', @next_cluster_number, '),('));
SET @values_to_insert = CONCAT('(', @values_to_insert, ', ', @next_cluster_number, ')'); -- This produces a string like this -> (<CONCEPT ID 1>, <CLUSTER ID>),(<CONCEPT ID 2>, <CLUSTER ID>),...

SET @insert_statement = CONCAT('INSERT INTO `graph_ontology`.`Edges_N_ConceptsCluster_N_Concept_T_ParentToChild` (`to_id`, `from_id`) VALUES', @values_to_insert); -- Build INSERT statement like this -> INSERT INTO RolesMenus VALUES(1, 100),(2, 100),(3, 100)

-- Execute INSERT statement
PREPARE stmt FROM @insert_statement;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Inserting the new cluster into the category-cluster table

INSERT INTO `graph_ontology`.`Edges_N_Category_N_ConceptsCluster_T_ParentToChild`
(`from_id`,
`to_id`)
VALUES
(@new_category_id,
@next_cluster_number);

-- Switching the is_ontology_concept flag on in the concepts table

UPDATE `graph_ontology`.`Nodes_N_Concept`
SET
`is_ontology_concept` = 1,
`is_ontology_neighbour` = 0,
`is_unused` = 0
WHERE FIND_IN_SET(id, @concepts_to_add) > 0;
