import json

import numpy as np
import pandas as pd

from db_cache_manager.db import DB

from graphai.core.common.config import config


class ConceptEmbeddings:
    """
    Class that holds the concept embeddings data for that vector computations
    """

    def __init__(self, concept_ids):
        self.concept_ids = concept_ids

        ################################################################

        db = DB(config['database'])

        ################################################################

        # Fetch concept embeddings
        table = 'graph.Embedding_Concepts_Ontology_Neighbour'
        fields = ['id', 'embedding']
        columns = ['concept_id', 'embedding']
        self.embeddings = pd.DataFrame(columns=columns)

        batch_size = 1000   # Fetch no more than 1000 concepts at once (otherwise the mysql request will time out because of the maximum query size of 4MB)
        for i in range(0, len(concept_ids), batch_size):
            conditions = {'id': concept_ids[i: i + batch_size]}
            self.embeddings = pd.concat([self.embeddings, pd.DataFrame(db.find(table_name=table, fields=fields, conditions=conditions), columns=columns)])

        # Drop concepts with null embeddings
        self.embeddings = self.embeddings.dropna().reset_index(drop=True)

        # Parse embeddings from string to list of floats
        self.embeddings['embedding'] = self.embeddings['embedding'].apply(json.loads)

        # Expand the column with the list of floats into multiple float columns (one per dimension)
        self.embeddings = pd.concat([self.embeddings, pd.DataFrame(self.embeddings['embedding'].tolist())], axis=1)

        # Drop the column with the list of floats as we don't need it anymore
        self.embeddings = self.embeddings.drop(columns=['embedding'])

        # Complete DataFrame with concepts in the original list that might not have embeddings
        self.embeddings = pd.merge(pd.Series(concept_ids, name='concept_id'), self.embeddings, how='left', on='concept_id').fillna(0)

    def values(self):
        return self.embeddings.drop(columns=['concept_id']).values

    def __repr__(self):
        return self.embeddings.__repr__()


def compute_embedding_scores(results):
    # Make sure concept ids are strings (otherwise the request will take a lot of time and will return nothing)
    results['concept_id'] = results['concept_id'].astype(str)

    # Extract ids of all concepts in the DataFrame
    concept_ids = list(results['concept_id'].drop_duplicates())
    n_concepts = len(concept_ids)

    # If there is only one concept, its embedding local and global scores are 1, and we return directly
    if n_concepts == 1:
        results['embedding_local_score'] = 1
        results['embedding_global_score'] = 1
        return results

    # Build data structure containing the embeddings vectors
    embeddings = ConceptEmbeddings(concept_ids)

    # Extract vector matrix
    vectors = embeddings.values()

    # Compute the scalar product of all vectors pairwise
    scalar_products = vectors @ vectors.T

    # Set negative scalar prods to 0. This is a hard threshold, but zero scalar product indicates that the two vectors are unrelated,
    # so we apply this same principle to vectors that are further away.
    scalar_products = np.maximum(scalar_products, 0)

    # Now scalar products are in [0, 1].
    # We apply an S-shaped transformation to the scalar products so that low values are lowered and high values are boosted.
    # The following function f: [0, 1] -> [0, 1] satisfies the following conditions:
    #   * f continuous (step-wise polynomial of degree n), differentiable in (0, 1) and increasing
    #   * f(0) = 0
    #   * f(1) = 1
    #   * f(a) = a
    #   * f'(x) -> 0 as x -> 0+
    #   * f'(x) -> 0 as x -> 1-
    #   * f'(a) = n
    #   * For n=1, f is the identity function
    #
    # TL;DR
    #   f is a smooth bijection of [0, 1], that lowers values before a, fixes a and increases values above a.
    #   The degree n controls how steep the function is around a: the higher the n, the more extreme the transformation is.
    def f(n, a, x):
        assert n >= 1
        assert 0 < a < 1

        x = min(1, max(0, x))

        if x <= a:
            return np.power(x, n) / np.power(a, n - 1)
        else:
            return 1 - np.power(1 - x, n) / np.power(1 - a, n - 1)

    n = 2
    a = 0.3
    scores = np.vectorize(f)(n, a, scalar_products)

    # Embedding local score is the mean scalar product among concepts with the same keywords
    embedding_local_scores = pd.DataFrame()
    for name, group in results.groupby('keywords'):
        # Extract concept_ids in the given keywords group and filter out the scores matrix with it
        group_concept_ids = list(group['concept_id'])
        n_group_concepts = len(group_concept_ids)

        # If there is only one concept in the group, its embedding local score is directly 1, and we skip to the next group
        if n_group_concepts == 1:
            group_embedding_local_scores = pd.DataFrame({'keywords': name, 'concept_id': group_concept_ids, 'embedding_local_score': 1})
            embedding_local_scores = pd.concat([embedding_local_scores, group_embedding_local_scores], ignore_index=True)
            continue

        # There are more than one concept, we proceed using the scalar products within the concepts of that keyword
        indices = [concept_id in group_concept_ids for concept_id in concept_ids]
        group_scores = scores[indices, :][:, indices]

        # Embedding local score is now the mean scalar product for this new matrix excluding the diagonal
        #   We create a n_group_concepts x n_group_concepts matrix with zeros in the diagonal and ones elsewhere.
        #   We use it as weights for the weighted average per row.
        #   Then we derive the score between 0 and 1 by mapping the cos to [0, 1] and clipping
        weights = np.ones((n_group_concepts, n_group_concepts)) - np.identity(n_group_concepts)
        group_embedding_local_scores = np.average(group_scores, axis=0, weights=weights)
        group_embedding_local_scores = pd.DataFrame({'keywords': name, 'concept_id': [concept_id for concept_id in concept_ids if concept_id in group_concept_ids], 'embedding_local_score': group_embedding_local_scores})
        embedding_local_scores = pd.concat([embedding_local_scores, group_embedding_local_scores], ignore_index=True)

    # Embedding global score is the mean scalar product excluding the diagonal
    #   We create a n_concepts x n_concepts matrix with zeros in the diagonal and ones elsewhere.
    #   We use it as weights for the weighted average per row.
    #   Then we derive the score between 0 and 1 by mapping the cos to [0, 1] and clipping
    weights = np.ones((n_concepts, n_concepts)) - np.identity(n_concepts)
    embedding_global_scores = np.average(scores, axis=0, weights=weights)
    embedding_global_scores = pd.DataFrame({'concept_id': concept_ids, 'embedding_global_score': embedding_global_scores})

    # Add the scores to the results DataFrame
    results = pd.merge(results, embedding_local_scores, how='inner', on=['keywords', 'concept_id'])
    results = pd.merge(results, embedding_global_scores, how='inner', on='concept_id')

    # Normalise scores in [0, 1]
    results['embedding_local_score'] = results['embedding_local_score'] / results['embedding_local_score'].max()
    results['embedding_global_score'] = results['embedding_global_score'] / results['embedding_global_score'].max()

    return results


if __name__ == '__main__':
    from elasticsearch_interface.es import ES

    from graphai.core.text.keywords import extract_keywords
    from graphai.core.text.wikisearch import wikisearch

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    es = ES(config['elasticsearch'], index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'))

    raw_text = """Consider a nonparametric representation of acoustic wave fields that consists of observing the sound pressure along a straight line or a smooth contour L defined in space. The observed data contains implicit information of the surrounding acoustic scene, both in terms of spatial arrangement of the sources and their respective temporal evolution. We show that such data can be effectively analyzed and processed in what we call the space-time-frequency representation space, consisting of a Gabor representation across the spatio-temporal manifold defined by the spatial axis L and the temporal axis t. In the presence of a source, the spectral patterns generated at L have a characteristic triangular shape that changes according to certain parameters, such as the source distance and direction, the number of sources, the concavity of L, and the analysis window size. Yet, in general, the wave fronts can be expressed as a function of elementary directional components-most notably, plane waves and far-field components. Furthermore, we address the problem of processing the wave field in discrete space and time, i.e., sampled along L and t, where a Gabor representation implies that the wave fronts are processed in a block-wise fashion. The key challenge is how to chose and customize a spatio-temporal filter bank such that it exploits the physical properties of the wave field while satisfying strict requirements such as perfect reconstruction, critical sampling, and computational efficiency. We discuss the architecture of such filter banks, and demonstrate their applicability in the context of real applications, such as spatial filtering, deconvolution, and wave field coding."""
    keyword_list = extract_keywords(raw_text)
    results = wikisearch(keyword_list, es)
    results['concept_id'] = results['concept_id'].astype(str)
    print(results)

    results = compute_embedding_scores(results)
    print(results)
