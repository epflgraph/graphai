import pandas as pd

import Levenshtein

from graphai.core.text.embeddings import compute_embedding_scores


def compute_levenshtein_score(results):
    # Compute levenshtein score
    results['levenshtein_score'] = results.apply(
        lambda row: Levenshtein.ratio(
            row['keywords'], row['concept_name'].replace('_', ' ').lower()
        ),
        axis=1,
    )

    # Smooth score using an S-shaped function on [0, 1] that pulls values away from 1/2, exaggerating differences
    def f(x):
        return 1 / (1 + ((1 - x) / x) ** 2)
    results['levenshtein_score'] = f(results['levenshtein_score'])

    return results


def compute_keywords_scores(results, smoothing):
    # Compute keywords scores aggregating other scores over keywords as an indicator for low-quality keywords
    # We compute three keywords scores: embedding_keywords_score (new), graph_keywords_score (new) and ontology_keywords_score (classical).

    results = pd.merge(
        results,
        results.groupby(by=['keywords']).aggregate(embedding_keywords_score=('embedding_global_score', 'sum')).reset_index(),
        how='left',
        on=['keywords']
    )

    results = pd.merge(
        results,
        results.groupby(by=['keywords']).aggregate(graph_keywords_score=('graph_score', 'sum')).reset_index(),
        how='left',
        on=['keywords']
    )

    results = pd.merge(
        results,
        results.groupby(by=['keywords']).aggregate(ontology_keywords_score=('ontology_global_score', 'sum')).reset_index(),
        how='left',
        on=['keywords']
    )

    # Normalise all three keywords scores to [0, 1]
    results['embedding_keywords_score'] = results['embedding_keywords_score'] / results['embedding_keywords_score'].max()
    results['graph_keywords_score'] = results['graph_keywords_score'] / results['graph_keywords_score'].max()
    results['ontology_keywords_score'] = results['ontology_keywords_score'] / results['ontology_keywords_score'].max()

    # Smooth scores if needed using the function f(x) = (2 - x) * x, bumping lower values to avoid very low scores
    if smoothing:
        results['embedding_keywords_score'] = (2 - results['embedding_keywords_score']) * results['embedding_keywords_score']
        results['graph_keywords_score'] = (2 - results['graph_keywords_score']) * results['graph_keywords_score']
        results['ontology_keywords_score'] = (2 - results['ontology_keywords_score']) * results['ontology_keywords_score']

    return results


def compute_mixed_score(results):
    # Compute mixed score as a convex combination of the different scores,
    # with prescribed coefficients found after running some analyses on manually tagged data.

    # Keep old coefficients for reference
    # old_coefficients = pd.DataFrame({
    #     'search_score': [0.2],
    #     'levenshtein_score': [0.15],
    #     'embedding_local_score': [0],
    #     'embedding_global_score': [0],
    #     'graph_score': [0.1],
    #     'ontology_local_score': [0.15],
    #     'ontology_global_score': [0.1],
    #     'embedding_keywords_score': [0],
    #     'graph_keywords_score': [0],
    #     'ontology_keywords_score': [0.3],
    # })

    # New coefficients after running benchmark for embeddings
    coefficients = pd.DataFrame({
        'search_score': [0.15],
        'levenshtein_score': [0.15],
        'embedding_local_score': [0.05],
        'embedding_global_score': [0.25],
        'graph_score': [0.05],
        'ontology_local_score': [0.1],
        'ontology_global_score': [0.1],
        'embedding_keywords_score': [0],
        'graph_keywords_score': [0],
        'ontology_keywords_score': [0.15],
    })
    results['mixed_score'] = results[coefficients.columns] @ coefficients.transpose()

    # Sort descendingly by mixed_score
    results = results.sort_values(by='mixed_score', ascending=False)

    return results


def aggregate_results(results, coef=0.5):
    """
    Aggregates a pandas DataFrame of keyword-concept results, i.e. unique by (keywords, concept_id), into a pandas DataFrame of concept results,
    i.e. unique by concept_id.

    Args:
        results (pd.DataFrame): A pandas DataFrame with columns ['keywords', 'concept_id', 'concept_name'] and a column 'x_score' for each score.
        coef (float): A number in [0, 1] that controls how the scores of the aggregated concepts are computed.
        A value of 0 takes the sum of scores over keywords, then normalises in [0, 1]. A value of 1 takes the max of scores over keywords.
        Any value in between linearly interpolates those two approaches. Default: 0.5.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['concept_id', 'concept_name'] and a column 'x_score' for each score.
    """

    ################################################################
    # Approach to aggregate results                                #
    ################################################################
    #
    # We need to aggregate results, which at this point are unique by (keywords, concept_id),
    # over keywords, so that they are unique by concept_id.
    #
    # To do so, several methods have been considered when grouping by concept_id:
    #   1. Take sum of scores.
    #   2. Take max of scores.
    #   3. Take median of scores.
    #   4. Take max of the first two values: 1. and 2.
    #   5. Take arithmetic mean of the first two values: 1. and 2.
    #   6. Take geometric mean of the first two values: 1. and 2.
    #   7. Take log(1 + sum of scores), then divide by maximum.
    #
    # All scores are divided by the maximum value to bring them back to [0, 1]. This is only a big concern for
    # option 1., for the rest it is either irrelevant or has limited impact.
    #
    # The following properties have been considered when choosing one method:
    #   A. If a concept only appears with one set of keywords, its final score cannot tend to zero as the number of
    #      sets of keywords increases.
    #   B. If a concept appears for $n$ sets of keywords with constant score $a$ and another page appears for $n$ sets of
    #      keywords with constant score $b$, such that $a < b < 1$, then the final score of the former has to be
    #      greater than $a$.
    #   C. If a concept appears for $n$ sets of keywords with constant score $a$ and another concept appears for $n+1$ sets
    #      of keywords with constant score $a$, then the final score of the latter must be strictly greater than
    #      the final score of the former.
    #   D. Let $C_1, C_2, C_3$ be three concepts appearing for $n$, $n+1$ and $n+2$ sets of keywords, respectively,
    #      in such a way that their scores are $\{x_1, ..., x_n\}$, $\{x_1, ..., x_n, a\}$ and
    #      $\{x_1, ..., x_n, a, a\}$, respectively. Then the difference of the final scores of $C_2$ and $C_1$ must be
    #      larger than the difference of the final scores of $C_3$ and $C_2$.
    #
    # None of the methods above satisfies all of these properties:
    #   * Property A is satisfied by 2., 3., 4. and 5., and violated by 1., 6. and 7.
    #   * Property B is satisfied by all methods.
    #   * Property C is satisfied by 1., 5., 6. and 7., and violated by 2., 3. and 4.
    #   * Property D is satisfied by 6. and 7., and violated by 1., 2., 3., 4. and 5.
    #
    # TL;DR
    # After considering this and running some tests, we decide to use a convex combination between methods 1 and 2,
    # tuned by a coefficient in [0, 1], in such a way that 0 gives method 1, 1 gives method 2 and 0.5 gives method 5.
    # The default is method 5.

    # Extract names not to group by them
    names = results[['concept_id', 'concept_name']].drop_duplicates(subset='concept_id')

    # Aggregate over pages, compute sum and max of scores
    results = results.groupby(by='concept_id').aggregate(
        search_score_sum=('search_score', 'sum'),
        search_score_max=('search_score', 'max'),
        levenshtein_score_sum=('levenshtein_score', 'sum'),
        levenshtein_score_max=('levenshtein_score', 'max'),
        embedding_local_score_sum=('embedding_local_score', 'sum'),
        embedding_local_score_max=('embedding_local_score', 'max'),
        embedding_global_score_sum=('embedding_global_score', 'sum'),
        embedding_global_score_max=('embedding_global_score', 'max'),
        graph_score_sum=('graph_score', 'sum'),
        graph_score_max=('graph_score', 'max'),
        ontology_local_score_sum=('ontology_local_score', 'sum'),
        ontology_local_score_max=('ontology_local_score', 'max'),
        ontology_global_score_sum=('ontology_global_score', 'sum'),
        ontology_global_score_max=('ontology_global_score', 'max'),
        embedding_keywords_score_sum=('embedding_keywords_score', 'sum'),
        embedding_keywords_score_max=('embedding_keywords_score', 'max'),
        graph_keywords_score_sum=('graph_keywords_score', 'sum'),
        graph_keywords_score_max=('graph_keywords_score', 'max'),
        ontology_keywords_score_sum=('ontology_keywords_score', 'sum'),
        ontology_keywords_score_max=('ontology_keywords_score', 'max'),
    ).reset_index()

    # Recover names after grouping
    results = pd.merge(results, names, how='left', on='concept_id')

    score_columns = [
        "search_score",
        "levenshtein_score",
        "embedding_local_score",
        "embedding_global_score",
        "graph_score",
        "ontology_local_score",
        "ontology_global_score",
        "embedding_keywords_score",
        "graph_keywords_score",
        "ontology_keywords_score",
    ]

    # Normalise sum scores to [0, 1]
    for column in score_columns:
        max_score_sum = results[f'{column}_sum'].max()
        if max_score_sum > 0:
            results[f'{column}_sum'] = results[f'{column}_sum'] / results[f'{column}_sum'].max()

    # Take convex combination of the two columns
    assert 0 <= coef <= 1, f'Normalisation coefficient {coef} is not in [0, 1]'
    for column in score_columns:
        results[column] = coef * results[f'{column}_sum'] + (1 - coef) * results[f'{column}_max']

    results = results[['concept_id', 'concept_name'] + score_columns]

    return results


def filter_results(results, epsilon=0.15):
    """
    Filters a DataFrame of concept results depending on their scores, based on some criteria specified in the parameters.

    Args:
        results (pd.DataFrame): A pandas DataFrame with the column 'mixed_score'.
        epsilon (float): A number in [0, 1] that is used as a threshold on 'mixed_score' to decide whether to keep the concept. Default: 0.15.

    Returns:
        pd.DataFrame: A DataFrame with the same columns as 'results' and a subset of its rows.
    """

    # Epsilon should be in [0, 1]
    assert 0 <= epsilon <= 1, f"Filtering threshold {epsilon} is not in [0, 1]"

    results = results[results['mixed_score'] >= epsilon]

    return results


def compute_scores(
    results,
    graph,
    restrict_to_ontology=False,
    score_smoothing=True,
    aggregation_coef=0.5,
    filtering_threshold=0.15,
    refresh_scores=True,
):
    """
    Gathers wikisearch results, computes several scores for them, and finally aggregates and filters them.

    Args:
        results (pd.DataFrame): A pandas DataFrame with columns ['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score'].
        graph (ConceptsGraph): The concepts graph and ontology object.
        restrict_to_ontology (bool): Whether to filter concepts that are not in the ontology. Default: False.
        score_smoothing (bool): Whether to apply a transformation to some scores to distribute them more evenly in [0, 1]. Default: True.
        aggregation_coef (float): A number in [0, 1] that controls how the scores of the aggregated pages are computed.
        A value of 0 takes the sum of scores over Keywords, then normalises in [0, 1]. A value of 1 takes the max of scores over Keywords.
        Any value in between linearly interpolates those two approaches. Default: 0.5.
        filtering_threshold (float): A number in [0, 1] that is used as a threshold for all the scores to decide whether the page is good enough
        from that score's perspective. Default: 0.15.
        refresh_scores (bool): Whether to recompute scores after filtering. Default: True.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['concept_id', 'concept_name'] and a column 'x_score' for each score,
        including a 'mixed_score' with a weighted average of the other scores.
    """

    # Return if there are no results
    if len(results) == 0:
        return results

    # Parse concept id type from int to string
    results['concept_id'] = results['concept_id'].astype(str)

    # Restrict to ontology if needed
    if restrict_to_ontology:
        results = results[results['concept_id'].isin(graph.get_ontology_concepts())]

    # Compute levenshtein score
    results = compute_levenshtein_score(results)

    # Compute embeddings (local and global) scores
    results = compute_embedding_scores(results)

    # Compute graph score
    results = graph.add_graph_score(results, smoothing=score_smoothing)

    # Compute ontology (local and global) scores
    results = graph.add_ontology_scores(results, smoothing=score_smoothing)

    # Compute keywords scores
    results = compute_keywords_scores(results, smoothing=score_smoothing)

    # Return if there are no results
    if len(results) == 0:
        return results

    # Aggregate results over keywords to obtain one row per concept
    aggregated_results = aggregate_results(results, coef=aggregation_coef)

    # Compute mixed score
    aggregated_results = compute_mixed_score(aggregated_results)

    # Filter results
    aggregated_results = filter_results(aggregated_results, epsilon=filtering_threshold)

    # Return if there are no results
    if len(aggregated_results) == 0:
        return pd.DataFrame()

    # If scores don't need to be recomputed, compute mixed score and return
    if not refresh_scores:
        return aggregated_results

    # To recompute scores, we keep only the relevant unaggregated results that survive the aggregation and filtering,
    # we keep only the initial columns, as the rest need to be recomputed, and call the function again.
    results = pd.merge(results, aggregated_results['concept_id'], how='inner', on='concept_id')
    results = results[['keywords', 'concept_id', 'concept_name', 'search_score']]

    return compute_scores(
        results,
        graph,
        restrict_to_ontology=restrict_to_ontology,
        score_smoothing=score_smoothing,
        aggregation_coef=aggregation_coef,
        filtering_threshold=0,
        refresh_scores=False,
    )


if __name__ == '__main__':
    from elasticsearch_interface.es import ESConceptDetection

    from graphai.core.common.config import config

    from graphai.core.text.keywords import extract_keywords
    from graphai.core.text.wikisearch import wikisearch
    from graphai.core.text.graph import ConceptsGraph

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    es = ESConceptDetection(config['elasticsearch'],
                            index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'))
    graph = ConceptsGraph()

    raw_text = """Consider a nonparametric representation of acoustic wave fields that consists of observing the sound pressure along a straight line or a smooth contour L defined in space. The observed data contains implicit information of the surrounding acoustic scene, both in terms of spatial arrangement of the sources and their respective temporal evolution. We show that such data can be effectively analyzed and processed in what we call the space-time-frequency representation space, consisting of a Gabor representation across the spatio-temporal manifold defined by the spatial axis L and the temporal axis t. In the presence of a source, the spectral patterns generated at L have a characteristic triangular shape that changes according to certain parameters, such as the source distance and direction, the number of sources, the concavity of L, and the analysis window size. Yet, in general, the wave fronts can be expressed as a function of elementary directional components-most notably, plane waves and far-field components. Furthermore, we address the problem of processing the wave field in discrete space and time, i.e., sampled along L and t, where a Gabor representation implies that the wave fronts are processed in a block-wise fashion. The key challenge is how to chose and customize a spatio-temporal filter bank such that it exploits the physical properties of the wave field while satisfying strict requirements such as perfect reconstruction, critical sampling, and computational efficiency. We discuss the architecture of such filter banks, and demonstrate their applicability in the context of real applications, such as spatial filtering, deconvolution, and wave field coding."""
    keyword_list = extract_keywords(raw_text)
    results = wikisearch(keyword_list, es)
    results['concept_id'] = results['concept_id'].astype(str)
    print(results)

    results = compute_scores(results, graph)
    print(results)
