-- Changing the flag of the concept

UPDATE graph_ontology.Nodes_N_Concept 
SET is_ontology_neighbour=0,
is_ontology_concept=0,
is_ontology_category=1
WHERE id='55955';

-- Inserting the depth-4 category

INSERT INTO graph_ontology.Nodes_N_Category 
(`id`, `name`, `depth`, `reference_page_id`, `reference_page_key`, `reference_page_url`)
VALUES
('version-control', 'Version control', 4, '55955', 'Version_control', 'https://en.wikipedia.org/wiki/Version_control');

-- Inserting the anchor pages

INSERT INTO graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage (`from_id`, `to_id`)
VALUES
('version-control', '55955');

-- Inserting the child to parent relationship

INSERT INTO graph_ontology.Edges_N_Category_N_Category_T_ChildToParent (`from_id`, `to_id`)
VALUES
('version-control', 'software-development');

-- Deleting the anchor pages of the new categories from the cluster-concept table

DELETE FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild
WHERE to_id='55955';

