import numpy as np
import pandas as pd

from interfaces.db import DB


class ConceptsGraph:
    def __init__(self):
        db = DB()

        table_name = 'graph.Nodes_N_Concept'
        fields = ['PageID', 'PageTitle']
        self.concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
        concept_ids = list(self.concepts['PageID'])

        table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
        fields = ['SourcePageID', 'TargetPageID']
        conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
        concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

        # Store successor and predecessor sets as attributes
        self.successors = concepts_concepts.groupby(by='SourcePageID').aggregate({'TargetPageID': set})
        self.predecessors = concepts_concepts.groupby(by='TargetPageID').aggregate({'SourcePageID': set})
        self.sources = set(self.successors.index)
        self.targets = set(self.predecessors.index)

    def compute_scores(self, source_page_ids, target_page_ids):
        """
        Computes the graph scores for all possible source-target pairs from two lists of page ids.
        The graph score of a pair (s, t) is computed as:

        :math:`\\displaystyle score(s, t) = 1 - \\frac{1}{1 + \\ln(1 + rebound(s, t) * \\ln(out(s, t)))}`, with

        :math:`\\displaystyle rebound(s, t) = \\frac{1 + \\min\\{in(s, t), out(s, t)\\}}{1 + \max\\{in(s, t), out(s, t)\\}}`,

        :math:`in(s, t) =` number of paths from t to s,

        :math:`out(s, t) =` number of paths from s to t.

        Args:
            source_page_ids (list[int]): List of source page ids.
            target_page_ids (list[int]): List of target page ids.

        Returns:
            list[dict[str]]: A list with all possible source-target pairs and their graph score. Each element of the list
            has keys 'source_page_id' (int), 'target_page_id' (int) and 'score' (float).

        Examples:
            >>> cg = ConceptsGraph()
            >>> cg.compute_scores([6220, 1196], [18973446, 9417, 946975])
            [{'source_page_id': 6220, 'target_page_id': 18973446, 'score': 0.5193576613841849}, {'source_page_id': 6220, 'target_page_id': 9417, 'score': 0.4740698357575134}, {'source_page_id': 6220, 'target_page_id': 946975, 'score': 0.3591100561928845}, {'source_page_id': 1196, 'target_page_id': 18973446, 'score': 0.5343247664351338}, {'source_page_id': 1196, 'target_page_id': 9417, 'score': 0.530184349205493}, {'source_page_id': 1196, 'target_page_id': 946975, 'score': 0.40043881227279876}]
        """
        pairs = [(s, t) for s in source_page_ids for t in target_page_ids]

        results = []
        for s, t in pairs:

            # If both pages are the same, then the score is 1
            if s == t:
                results.append({
                    'source_page_id': s,
                    'target_page_id': t,
                    'score': 1
                })
                continue

            # If s has no successors or t has no predecessors, then there are no outgoing paths. The score is 0.
            if (s not in self.sources) or (t not in self.targets):
                results.append({
                    'source_page_id': s,
                    'target_page_id': t,
                    'score': 0
                })
                continue

            # Compute the number of outgoing paths of length <= 2
            s_out = set(self.successors.loc[s, 'TargetPageID']) - {s}
            t_in = set(self.predecessors.loc[t, 'SourcePageID']) - {t}
            n_out_paths = len(s_out & t_in) + (1 if t in s_out else 0)

            # If there are no outgoing paths, the score is zero
            if n_out_paths == 0:
                results.append({
                    'source_page_id': s,
                    'target_page_id': t,
                    'score': 0
                })
                continue

            # Compute the number of ingoing paths of length <= 2
            if (s not in self.targets) or (t not in self.sources):
                # If s has no predecessors or t has no successors, then there are no ingoing paths.
                n_in_paths = 0
            else:
                s_in = set(self.predecessors.loc[s, 'SourcePageID']) - {s}
                t_out = set(self.successors.loc[s, 'TargetPageID']) - {t}
                n_in_paths = len(s_in & t_out) + (1 if s in t_out else 0)

            # Compute score
            rebound = (1 + min(n_out_paths, n_in_paths)) / (1 + max(n_out_paths, n_in_paths))
            score = 1 - 1 / (1 + np.log(1 + rebound * np.log(n_out_paths)))

            # Append result
            results.append({
                'source_page_id': s,
                'target_page_id': t,
                'score': score
            })

        return results


class Ontology:
    def __init__(self):
        db = DB()

        ################################################

        # Fetch cluster nodes
        table_name = 'graphontology.Hierarchical_Cluster_Names_HighLevel'
        fields = ['ClusterIndex', 'ClusterName']
        self.clusters = pd.DataFrame(db.find(table_name, fields=fields), columns=['ClusterID', 'ClusterName'])

        # Extract cluster ids for faster access
        self.cluster_ids = set(self.clusters['ClusterID'])

        # Store cluster names indexing by ClusterID for faster access
        self.cluster_names = self.clusters.set_index('ClusterID')['ClusterName']

        ################################################

        # Fetch cluster-cluster edges
        table_name = 'graphontology.Predefined_Knowledge_Tree_Hierarchy'
        fields = ['ChildIndex', 'ParentIndex']
        self.clusters_clusters = pd.DataFrame(db.find(table_name, fields=fields), columns=['ChildID', 'ParentID'])

        # Extract child cluster ids for faster access
        self.cluster_child_ids = set(self.clusters_clusters['ChildID'])

        # Store cluster parents indexing by ChildID for faster access
        self.cluster_parents = self.clusters_clusters.set_index('ChildID')['ParentID']

        ################################################

        # Fetch concept-cluster edges
        table_name = 'graphontology.Hierarchical_Clusters_Main'
        fields = ['PageID', 'ClusterLevel1']
        self.concepts_clusters = pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID', 'ClusterID'])

        # Extract concept and cluster ids for faster access
        self.concept_ids = set(self.concepts_clusters['PageID'])

        # Store cluster parents indexing by ChildID for faster access
        self.concept_clusters = self.concepts_clusters.set_index('PageID')['ClusterID']

    def get_concept_cluster(self, page_id):
        if page_id not in self.concept_ids:
            # print(f"""Warning! Concept {page_id} doesn't have a cluster in the ontology""")
            return None

        return self.concept_clusters.at[page_id]

    def add_cluster(self, results):
        return pd.merge(results, self.concepts_clusters, how='inner', on='PageID')
