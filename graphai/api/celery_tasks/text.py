import pandas as pd
from time import sleep

from celery import shared_task

import Levenshtein

from elasticsearch_interface.es import ES

from graphai.api.common.graph import graph
from graphai.api.common.ontology import ontology

from graphai.core.common.config import config
from graphai.core.common.common_utils import strtobool

from graphai.core.interfaces.wp import WP

from graphai.core.text.keywords import get_keywords
from graphai.core.text.draw import draw_ontology, draw_graph


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.extract_keywords', ignore_result=False)
def extract_keywords_task(self, raw_text, use_nltk=False):
    """
    Celery task that extracts keywords from a given text.

    Args:
        raw_text (str|list): Text to extract keywords from. If a list is passed, it is assumed that they are the keywords and the same list is returned.
        use_nltk (bool): Whether to use nltk-rake for keyword extraction, otherwise python-rake is used. Default: False.

    Returns:
        list[str]: A list containing the keywords extracted from the text.
    """

    if isinstance(raw_text, list):
        return raw_text

    keywords_list = get_keywords(raw_text, use_nltk)

    return keywords_list


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.wikisearch', ignore_result=False, wp=WP(), es=ES(config['elasticsearch'], 'aitor_concepts'))
def wikisearch_task(self, keywords_list, fraction=(0, 1), method='es-base'):
    """
    Celery task that finds 10 relevant Wikipedia pages for each set of keywords in a list.

    Args:
        keywords_list (list(str)): List containing the sets of keywords for which to search Wikipedia pages.
        fraction (tuple(int, int)): Portion of the keywords_list to be processed, e.g. (1/3, 2/3) means only
        the middle third of the list is considered.
        method (str): Method to retrieve the wikipedia pages. It can be either "wikipedia-api", to use the
        Wikipedia API, or one of {"es-base", "es-score"}, to use elasticsearch, returning as score the inverse
        of the searchrank or the actual elasticsearch score, respectively. Default: 'es-base'. Fallback:
        'wikipedia-api'.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore', 'LevenshteinScore'],
        constant on 'Keywords' and unique in 'PageID', with the wikisearch results the given keywords set.
    """

    # Slice keywords_list
    begin = int(fraction[0] * len(keywords_list))
    end = int(fraction[1] * len(keywords_list))
    keywords_list = keywords_list[begin:end]

    # Iterate over all keyword sets and request the results
    all_results = pd.DataFrame()
    for keywords in keywords_list:
        # Request results
        if method == 'wikipedia-api':
            results = self.wp.search(keywords)
        else:
            # Try to get elasticsearch results, fallback to Wikipedia API in case of error.
            try:
                hits = self.es.search(keywords)

                results = pd.DataFrame(
                    [
                        [
                            hits[i]['_source']['id'],
                            hits[i]['_source']['title'],
                            (i + 1),
                            hits[i]['_score']
                        ]
                        for i in range(len(hits))
                    ],
                    columns=['PageID', 'PageTitle', 'Searchrank', 'SearchScore']
                )
            except Exception as e:
                print('[ERROR] Error connecting to elasticsearch cluster. Falling back to Wikipedia API.')
                print(e)
                results = self.wp.search(keywords)

            # Fallback to Wikipedia API if no results from elasticsearch
            if len(results) == 0:
                print(f'[WARNING] No results from elasticsearch cluster for keywords {keywords}. Falling back to Wikipedia API.')
                results = self.wp.search(keywords)

        # Ignore set of keywords if no pages are found
        if len(results) == 0:
            continue

        # Replace score with linear function on Searchrank if needed
        if method != 'es-score':
            results['SearchScore'] = 1 - (results['Searchrank'] - 1) / len(results)

        # Add Keywords column
        results['Keywords'] = keywords

        # Compute levenshtein score
        # S-shaped function on [0, 1] that pulls values away from 1/2, exaggerating differences
        def f(x):
            return 1 / (1 + ((1 - x) / x) ** 2)

        results['LevenshteinScore'] = results.apply(
            lambda row: Levenshtein.ratio(keywords, row['PageTitle'].replace('_', ' ').lower()), axis=1)
        results['LevenshteinScore'] = f(results['LevenshteinScore'])

        # Rearrange columns and concatenate with all results
        results = results[['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore', 'LevenshteinScore']]
        all_results = pd.concat([all_results, results], ignore_index=True)

    return all_results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.wikisearch_callback', ignore_result=False)
def wikisearch_callback_task(self, results):
    """
    Celery task that aggregates the results from the wikisearch task. It combines all parallel results into one single DataFrame.

    Args:
        results (list(pd.DataFrame)): List of DataFrames with columns ['Keywords', 'PageID', 'PageTitle', 'Searchrank',
        'SearchScore', 'LevenshteinScore'].

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore', 'LevenshteinScore'].
    """

    # Concatenate all results in a single DataFrame
    results = pd.concat(results, ignore_index=True)

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.compute_scores', ignore_result=False, graph=graph, ontology=ontology)
def compute_scores_task(self, results, restrict_to_ontology=False, graph_score_smoothing=True, ontology_score_smoothing=True, keywords_score_smoothing=True):
    """
    Celery task that computes the GraphScore, OntologyLocalScore, OntologyGlobalScore and KeywordsScore to a DataFrame of wikisearch results.

    Args:
        results (pd.DataFrame): A pandas DataFrame containing the results of a wikisearch.
        It must contain the columns ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore'].
        restrict_to_ontology (bool): Whether to filter Wikipedia pages that are not in the ontology. Default: False.
        graph_score_smoothing (bool): Whether to apply a transformation to the GraphScore that bumps scores to avoid
        a negative exponential shape. Default: True.
        ontology_score_smoothing (bool): Whether to apply a transformation to the ontology scores that pushes scores away from 0.5. Default: True.
        keywords_score_smoothing (bool): Whether to apply a transformation to the KeywordsScore that bumps scores to avoid
        a negative exponential shape. Default: True.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore',
        'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore'].
    """

    if len(results) == 0:
        return results

    if restrict_to_ontology:
        results = self.ontology.filter_concepts(results)

    # Compute GraphScore
    results = self.graph.add_graph_score(results, smoothing=graph_score_smoothing)

    # Compute OntologyLocalScore and OntologyGlobalScore
    self.ontology.graph = self.graph
    results = self.ontology.add_ontology_scores(results, smoothing=ontology_score_smoothing)
    self.ontology.graph = None

    # Compute KeywordsScore aggregating OntologyGlobalScore over Keywords as an indicator for low-quality keywords
    results = pd.merge(
        results,
        results.groupby(by=['Keywords']).aggregate(KeywordsScore=('OntologyGlobalScore', 'sum')).reset_index(),
        how='left',
        on=['Keywords']
    )

    # Normalise the KeywordsScore to [0, 1]
    results['KeywordsScore'] = results['KeywordsScore'] / results['KeywordsScore'].max()

    # Smooth score if needed using the function f(x) = (2 - x) * x, bumping lower values to avoid very low scores
    if keywords_score_smoothing:
        results['KeywordsScore'] = (2 - results['KeywordsScore']) * results['KeywordsScore']

    return results


def aggregate_results(results, coef=0.5):
    """
    Aggregates a pandas DataFrame of intermediate wikify results, unique by (Keywords, PageID), into a pandas DataFrame
    of final wikify results, unique by PageID. Then computes the MixedScore for every page as a convex combination of the other scores.

    Args:
        results (pd.DataFrame): A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore',
        'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore'].
        coef (float): A number in [0, 1] that controls how the scores of the aggregated pages are computed.
        A value of 0 takes the sum of scores over Keywords, then normalises in [0, 1]. A value of 1 takes the max of scores over Keywords.
        Any value in between linearly interpolates those two approaches. Default: 0.5.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore',
        'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore'].
    """

    if len(results) == 0:
        return results

    ################################################################
    # Normalisation of results                                     #
    ################################################################

    # We need to aggregate results, which at this point are unique by (Keywords, PageID), over Keywords, so that they
    # are unique by PageID.
    #
    # To do so, several methods have been considered when grouping by PageID:
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
    #   A. If a page only appears with one set of keywords, its final score cannot tend to zero as the number of
    #      sets of keywords increases.
    #   B. If a page appears for $n$ sets of keywords with constant score $a$ and another page appears for $n$ sets of
    #      keywords with constant score $b$, such that $a < b < 1$, then the final score of the former has to be
    #      greater than $a$.
    #   C. If a page appears for $n$ sets of keywords with constant score $a$ and another page appears for $n+1$ sets
    #      of keywords with constant score $a$, then the final score of the latter must be strictly greater than
    #      the final score of the former.
    #   D. Let $P_1, P_2, P_3$ be three pages appearing for $n$, $n+1$ and $n+2$ sets of keywords, respectively,
    #      in such a way that their scores are $\{x_1, ..., x_n\}$, $\{x_1, ..., x_n, a\}$ and
    #      $\{x_1, ..., x_n, a, a\}$, respectively. Then the difference of the final scores of $P_2$ and $P-1$ must be
    #      larger than the difference of the final scores of $P_3$ and $P_2$.
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

    # Extract titles not to group by them
    titles = results[['PageID', 'PageTitle']].drop_duplicates(subset='PageID')

    # Aggregate over pages, compute sum and max of scores
    results = results.groupby(by='PageID').aggregate(
        SearchScoreSum=('SearchScore', 'sum'),
        SearchScoreMax=('SearchScore', 'max'),
        LevenshteinScoreSum=('LevenshteinScore', 'sum'),
        LevenshteinScoreMax=('LevenshteinScore', 'max'),
        GraphScoreSum=('GraphScore', 'sum'),
        GraphScoreMax=('GraphScore', 'max'),
        OntologyLocalScoreSum=('OntologyLocalScore', 'sum'),
        OntologyLocalScoreMax=('OntologyLocalScore', 'max'),
        OntologyGlobalScoreSum=('OntologyGlobalScore', 'sum'),
        OntologyGlobalScoreMax=('OntologyGlobalScore', 'max'),
        KeywordsScoreSum=('KeywordsScore', 'sum'),
        KeywordsScoreMax=('KeywordsScore', 'max')
    ).reset_index()

    # Recover titles from before grouping
    results = pd.merge(results, titles, how='left', on='PageID')

    score_columns = ['SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore']

    # Normalise sum scores to [0, 1]
    for column in score_columns:
        max_score_sum = results[f'{column}Sum'].max()
        if max_score_sum > 0:
            results[f'{column}Sum'] = results[f'{column}Sum'] / results[f'{column}Sum'].max()

    # Take convex combination of the two columns
    assert 0 <= coef <= 1, f'Normalisation coefficient {coef} is not in [0, 1]'
    for column in score_columns:
        results[column] = coef * results[f'{column}Sum'] + (1 - coef) * results[f'{column}Max']

    results = results[['PageID', 'PageTitle'] + score_columns]

    ################################################################
    # Compute mixed score                                          #
    ################################################################

    # Compute mixed score as a convex combination of the different scores,
    # with prescribed coefficients found after running some analyses on manually tagged data.
    coefficients = pd.DataFrame({
        'SearchScore': [0.2],
        'LevenshteinScore': [0.15],
        'GraphScore': [0.1],
        'OntologyLocalScore': [0.15],
        'OntologyGlobalScore': [0.1],
        'KeywordsScore': [0.3]
    })
    results['MixedScore'] = results[score_columns] @ coefficients.transpose()

    return results


def filter_results(results, epsilon=0.1, min_votes=5):
    """
    Filters a DataFrame of aggregated wikify results depending on their scores, based on some criteria specified in the parameters.

    Args:
        results (pd.DataFrame): A pandas DataFrame with columns ['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore',
        'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore'].
        epsilon (float): A number in [0, 1] that is used as a threshold for all the scores to decide whether the page is good enough
        from that score's perspective. Default: 0.1.
        min_votes (int): A number between 0 and the number of scores (excluding MixedScore). A page will be kept iff it is good enough
        for at least this amount of scores. Default: 5.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore',
        'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore'].
    """

    if len(results) == 0:
        return results

    ################################################################
    # Filtering of results                                         #
    ################################################################

    # Filter results with low scores through a majority vote among all scores
    # To be kept, we require a concept to have at least min_votes out of 6 scores to be significant (>= epsilon)
    assert 0 <= epsilon <= 1, f'Filtering threshold {epsilon} is not in [0, 1]'
    assert 0 <= min_votes <= 6, f'Filtering minimum number of votes {min_votes} is not in [0, 6]'

    votes = pd.DataFrame()
    score_columns = ['SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore']
    for column in score_columns:
        votes[column] = (results[column] >= epsilon).astype(int)
    votes = votes.sum(axis=1)
    results = results[votes >= min_votes]

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.purge_irrelevant', ignore_result=False)
def purge_irrelevant_task(self, results, coef=0.5, epsilon=0.1, min_votes=5):
    """
    Celery task that aggregates intermediate wikify results and filters irrelevant pages,
    then drops all scores that don't apply anymore for recomputation.

    Args:
        results (pd.DataFrame): A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore',
        'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore'].
        coef (float): A number in [0, 1] that controls how the scores of the aggregated pages are computed.
        A value of 0 takes the sum of scores over Keywords, then normalises in [0, 1]. A value of 1 takes the max of scores over Keywords.
        Any value in between linearly interpolates those two approaches. Default: 0.5.
        epsilon (float): A number in [0, 1] that is used as a threshold for all the scores to decide whether the page is good enough
        from that score's perspective. Default: 0.1.
        min_votes (int): A number between 0 and the number of scores (excluding MixedScore). A page will be kept iff it is good enough
        for at least this amount of scores. Default: 5.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore'].
    """

    # Aggregate and filter to know which pages are relevant
    relevant = filter_results(aggregate_results(results, coef=coef), epsilon=epsilon, min_votes=min_votes)

    if len(relevant) == 0:
        return pd.DataFrame()

    # Keep only relevant results, namely those whose PageID survives the aggregate_and_filter operation
    results = pd.merge(results, relevant['PageID'], how='inner', on='PageID')

    # Keep only SearchScore and LevenshteinScore, the rest need to be recomputed
    results = results[['Keywords', 'PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore']]

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.aggregate', ignore_result=False)
def aggregate_task(self, results, coef=0.5, filter=False, epsilon=0.1, min_votes=5):
    """
    Celery task that aggregates a pandas DataFrame of intermediate wikify results, unique by (Keywords, PageID), into a pandas DataFrame
    of final wikify results, unique by PageID. Then computes the MixedScore for every page as a convex combination of the other scores.
    Finally, it optionally filters pages depending on their scores, based on some criteria specified in the parameters.

    Args:
        results (pd.DataFrame): A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore',
        'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore'].
        coef (float): A number in [0, 1] that controls how the scores of the aggregated pages are computed.
        A value of 0 takes the sum of scores over Keywords, then normalises in [0, 1]. A value of 1 takes the max of scores over Keywords.
        Any value in between linearly interpolates those two approaches. Default: 0.5.
        filter (bool): Whether to filter any page. Default: False.
        epsilon (float): A number in [0, 1] that is used as a threshold for all the scores to decide whether the page is good enough
        from that score's perspective. Default: 0.1.
        min_votes (int): A number between 0 and the number of scores (excluding MixedScore). A page will be kept iff it is good enough
        for at least this amount of scores. Default: 5.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore',
        'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore'].
    """

    results = aggregate_results(results, coef=coef)

    if filter:
        results = filter_results(results, epsilon=epsilon, min_votes=min_votes)

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.draw_ontology', ignore_result=False, ontology=ontology)
def draw_ontology_task(self, results, level=2):
    """
    Celery task that draws the ontology neighbourhood induced by the given set of wikify results.

    Args:
        results (list(dict)): A serialised (orient='records') pandas DataFrame with columns ['PageID', 'PageTitle', 'SearchScore',
        'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore'].
        level (int): How many levels to go up in the ontology from the concepts. Default: 2.

    Returns:
        bool: Whether the drawing succeeded.
    """

    return draw_ontology(results, self.ontology, level)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.draw_graph', ignore_result=False, graph=graph)
def draw_graph_task(self, results, concept_score_threshold=0.3, edge_threshold=0.3, min_component_size=3):
    """
    Celery task that draws the concepts graph neighbourhood induced by the given set of wikify results.

    Args:
        results (list(dict)): A serialised (orient='records') pandas DataFrame with columns ['PageID', 'PageTitle', 'SearchScore',
        'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore'].
        concept_score_threshold (float): Score threshold below which concepts are filtered out. Default: 0.3.
        edge_threshold (float): Score threshold below which edges are filtered out. Default: 0.3.
        min_component_size (int): Size threshold below which connected components are filtered out. Default: 3.

    Returns:
        bool: Whether the drawing succeeded.
    """

    return draw_graph(results, self.graph, concept_score_threshold, edge_threshold, min_component_size)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.sleeper', ignore_result=False)
def text_test_task(self):
    sleep(15)
    print('it worked')
    return 0


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.init', ignore_result=False, graph=graph, ontology=ontology)
def text_init_task(self):
    """
    Celery task that spawns and populates graph and ontology objects so that they are held in memory ready for requests to arrive.
    """

    # This task initialises the text celery worker by loading into memory the graph and ontology tables
    print('Start text_init task')

    if strtobool(config['preload']['text']):
        print('Loading graph tables...')
        self.graph.fetch_from_db()

        print('Loading ontology tables...')
        self.ontology.fetch_from_db()
    else:
        print('Skipping preloading for text endpoints.')

    print('Graph and ontology tables loaded')
    return True
