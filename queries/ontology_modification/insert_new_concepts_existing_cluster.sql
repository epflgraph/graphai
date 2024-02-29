-- When the cluster already exists, all we have to do is to insert the appropriate cluster-concept rows
-- and to set the right flags for the newly-added concepts in the concepts table. There is no category id
-- in these queries because the category of the destination cluster will not change.

-- Setting the list of concepts and the id of the destination cluster

SET @concepts_to_add := '<CONCEPT ID 1>,<CONCEPT ID 2>,...';
SET @cluster_id := '<CLUSTER ID>';

-- Adding the concepts to the cluster-concept table

SET @values_to_insert = REPLACE(@concepts_to_add, ',', CONCAT(', ', @cluster_id, '),('));
SET @values_to_insert = CONCAT('(', @values_to_insert, ', ', @cluster_id, ')'); -- This produces a string like this -> (<CONCEPT ID 1>, <CLUSTER ID>),(<CONCEPT ID 2>, <CLUSTER ID>),...

SET @insert_statement = CONCAT('INSERT INTO `graph_ontology`.`Edges_N_ConceptsCluster_N_Concept_T_ParentToChild` (`to_id`, `from_id`) VALUES', @values_to_insert); -- Build INSERT statement like this -> INSERT INTO RolesMenus VALUES(1, 100),(2, 100),(3, 100)

-- Execute INSERT statement
PREPARE stmt FROM @insert_statement;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Switching the is_ontology_concept flag on in the concepts table

UPDATE `graph_ontology`.`Nodes_N_Concept`
SET
`is_ontology_concept` = 1,
`is_ontology_neighbour` = 0,
`is_unused` = 0
WHERE FIND_IN_SET(id, @concepts_to_add) > 0;
