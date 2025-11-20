-- Changing the flag of the concept

UPDATE graph_ontology.Nodes_N_Concept 
SET is_ontology_concept=0,
is_ontology_category=1
WHERE id='23204';

-- Inserting the depth-3 and depth-4 categories

INSERT INTO graph_ontology.Nodes_N_Category 
(`id`, `name`, `depth`, `reference_page_id`, `reference_page_key`, `reference_page_url`)
VALUES
('physical-quantities', 'Physical quantities', 3, '23204', 'Physical_quantity', 'https://en.wikipedia.org/wiki/Physical_quantity');

INSERT INTO graph_ontology.Nodes_N_Category 
(`id`, `name`, `depth`, `reference_page_id`, `reference_page_key`, `reference_page_url`)
VALUES
('topics-in-physical-quantities', 'Topics in physical quantities', 4, '23204', 'Physical_quantity', 'https://en.wikipedia.org/wiki/Physical_quantity');

INSERT INTO graph_ontology.Nodes_N_Category 
(`id`, `name`, `depth`, `reference_page_id`, `reference_page_key`, `reference_page_url`)
VALUES
('entities-in-physical-quantities', 'Entities in physical quantities', 4, '23204', 'Physical_quantity', 'https://en.wikipedia.org/wiki/Physical_quantity');

-- Inserting the anchor pages for all new categories

INSERT INTO graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage (`from_id`, `to_id`)
VALUES
('physical-quantities', '23204');

INSERT INTO graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage (`from_id`, `to_id`)
VALUES
('topics-in-physical-quantities', '23204');

INSERT INTO graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage (`from_id`, `to_id`)
VALUES
('entities-in-physical-quantities', '23204');

-- Inserting the child to parent relationships for all the new categories

INSERT INTO graph_ontology.Edges_N_Category_N_Category_T_ChildToParent (`from_id`, `to_id`)
VALUES
('physical-quantities', 'physics');

INSERT INTO graph_ontology.Edges_N_Category_N_Category_T_ChildToParent (`from_id`, `to_id`)
VALUES
('topics-in-physical-quantities', 'physical-quantities');

INSERT INTO graph_ontology.Edges_N_Category_N_Category_T_ChildToParent (`from_id`, `to_id`)
VALUES
('entities-in-physical-quantities', 'physical-quantities');

-- Deleting the anchor pages of the new categories from the cluster-concept table

DELETE FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild
WHERE to_id='23204';

