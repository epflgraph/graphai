-- Changing the flag of the concept

SET @d3_concept_id := '<DEPTH 3 ANCHOR CONCEPT ID>';
SET @d3_category_id := '<DEPTH 3 CATEGORY-ID>';
SET @d4_concept_id := '<DEPTH 4 ANCHOR CONCEPT ID>';
SET @d4_category_id := '<DEPTH 4 CATEGORY-ID>';
SET @d2_parent_id := '<DEPTH 2 PARENT CATEGORY-ID>';

UPDATE graph_ontology.Nodes_N_Concept
SET is_ontology_concept=0,
SET is_ontology_neighbour=0,
is_ontology_category=1
WHERE id=@d3_concept_id;

UPDATE graph_ontology.Nodes_N_Concept
SET is_ontology_concept=0,
SET is_ontology_neighbour=0,
is_ontology_category=1
WHERE id=@d4_concept_id;

-- Inserting the depth-3 and depth-4 categories

INSERT INTO graph_ontology.Nodes_N_Category
(`id`, `name`, `depth`, `reference_page_id`, `reference_page_key`, `reference_page_url`)
VALUES
(@d3_category_id, 'DEPTH 3 NAME', 3, @d3_concept_id, 'DEPTH 3 REF PAGE', 'DEPTH 3 REF PAGE URL');

INSERT INTO graph_ontology.Nodes_N_Category
(`id`, `name`, `depth`, `reference_page_id`, `reference_page_key`, `reference_page_url`)
VALUES
(@d4_category_id, 'DEPTH 4 NAME', 4, @d4_concept_id, 'DEPTH 4 REF PAGE', 'DEPTH 4 REF PAGE URL');

-- Inserting the anchor pages for all new categories

INSERT INTO graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage (`from_id`, `to_id`)
VALUES
(@d3_category_id, @d3_concept_id);

INSERT INTO graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage (`from_id`, `to_id`)
VALUES
(@d4_category_id, @d4_concept_id);

-- Inserting the child to parent relationships for all the new categories

INSERT INTO graph_ontology.Edges_N_Category_N_Category_T_ChildToParent (`from_id`, `to_id`)
VALUES
(@d3_category_id, @d2_parent_id);

INSERT INTO graph_ontology.Edges_N_Category_N_Category_T_ChildToParent (`from_id`, `to_id`)
VALUES
(@d4_category_id, @d3_category_id);

-- Deleting the anchor pages of the new categories from the cluster-concept table

DELETE FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild
WHERE to_id=@d3_concept_id OR to_id=@d4_concept_id;
