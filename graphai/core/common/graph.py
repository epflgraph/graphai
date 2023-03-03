import numpy as np
import pandas as pd

from core.interfaces.db import DB


class ConceptsGraph:
    def __init__(self):
        db = DB()

        table_name = 'graph.Nodes_N_Concept'
        fields = ['PageID', 'PageTitle']
        self.concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
        concept_ids = list(self.concepts['PageID'])

        table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
        fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
        conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
        self.concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

        self.concepts_concepts = pd.concat([
            self.concepts_concepts,
            self.concepts_concepts.rename(columns={'SourcePageID': 'TargetPageID', 'TargetPageID': 'SourcePageID'})
        ]).reset_index(drop=True)

        # Store successor and predecessor sets as attributes
        self.successors = self.concepts_concepts.groupby(by='SourcePageID').aggregate({'TargetPageID': set})
        self.predecessors = self.concepts_concepts.groupby(by='TargetPageID').aggregate({'SourcePageID': set})
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

    def add_graph_score(self, results):
        concept_ids = list(results['PageID'].drop_duplicates())
        concepts_concepts = self.concepts_concepts[
            self.concepts_concepts['SourcePageID'].isin(concept_ids) &
            self.concepts_concepts['TargetPageID'].isin(concept_ids)
        ]
        concepts_concepts = concepts_concepts.groupby(by='SourcePageID').aggregate(GraphScore=('NormalisedScore', 'sum')).reset_index()
        concepts_concepts['GraphScore'] = concepts_concepts['GraphScore'] / concepts_concepts['GraphScore'].max()
        concepts_concepts = concepts_concepts.rename(columns={'SourcePageID': 'PageID'})

        results = pd.merge(results, concepts_concepts, how='inner', on='PageID')

        return results
