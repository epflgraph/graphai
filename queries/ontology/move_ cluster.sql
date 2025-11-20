SET @cluster_id := '<CLUSTER ID>';
SET @category_id  := '<CATEGORY ID>';
UPDATE `graph_ontology`.`Edges_N_Category_N_ConceptsCluster_T_ParentToChild`
SET
`from_id` = @category_id
WHERE to_id = @cluster_id;
