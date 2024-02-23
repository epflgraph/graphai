-- If a group of concepts are to be split from their existing cluster and associated with a new category,
-- we need to calculate the next cluster id, change the cluster of those concepts to the new value,
-- and associate the new cluster with the target category.

-- Setting the name of the new category

SET @new_category_id := '<CATEGORY ID>';
SET @concepts_to_split := '<CONCEPT ID 1>,<CONCEPT ID 2>,...';

-- Computing the id of the new cluster

SET @next_cluster_number := (SELECT CAST(MAX(CAST(from_id AS UNSIGNED)) + 1 AS CHAR(255))
FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild
);

-- Updating the cluster value of the concepts and setting it to the new value

UPDATE `graph_ontology`.`Edges_N_ConceptsCluster_N_Concept_T_ParentToChild`
SET
`from_id` = @next_cluster_number
WHERE FIND_IN_SET(to_id, @concepts_to_split) > 0;

-- Inserting the new cluster into the category-cluster table

INSERT INTO `graph_ontology`.`Edges_N_Category_N_ConceptsCluster_T_ParentToChild`
(`from_id`,
`to_id`)
VALUES
(@new_category_id,
@next_cluster_number);

