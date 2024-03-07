SET @d4_concept_id := '<CONCEPT ID>';
SET @d4_category_id := '<NEW DEPTH 4 CATEGORY ID>';
SET @parent_category_id := '<PARENT CATEGORY ID>';

-- Changing the flag of the concept

UPDATE graph_ontology.Nodes_N_Concept
SET is_ontology_neighbour=0,
is_ontology_concept=0,
is_ontology_category=1
WHERE id=@d4_concept_id;

-- Inserting the depth-4 category

INSERT INTO graph_ontology.Nodes_N_Category
(`id`, `name`, `depth`, `reference_page_id`, `reference_page_key`, `reference_page_url`)
VALUES
(@d4_category_id, 'DEPTH 4 NAME', 4, @d4_concept_id, 'DEPTH 4 REF PAGE', 'DEPTH 4 REF URL');

-- Inserting the anchor pages

INSERT INTO graph_ontology.Edges_N_Category_N_Concept_T_AnchorPage (`from_id`, `to_id`)
VALUES
(@d4_category_id, @d4_concept_id);

-- Inserting the child to parent relationship

INSERT INTO graph_ontology.Edges_N_Category_N_Category_T_ChildToParent (`from_id`, `to_id`)
VALUES
(@d4_category_id, @parent_category_id);

-- Deleting the anchor pages of the new categories from the cluster-concept table

DELETE FROM graph_ontology.Edges_N_ConceptsCluster_N_Concept_T_ParentToChild
WHERE to_id=@d4_concept_id;
