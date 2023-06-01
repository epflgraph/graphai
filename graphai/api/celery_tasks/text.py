import pandas as pd
from time import sleep

from celery import shared_task

import Levenshtein

from graphai.api.common.graph import graph
from graphai.api.common.ontology import ontology

from graphai.core.interfaces.wp import WP
from graphai.core.interfaces.es import ES

from graphai.core.text.keywords import get_keywords


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.extract_keywords', ignore_result=False)
def extract_keywords_task(self, raw_text, use_nltk=False):
    keywords_list = get_keywords(raw_text, use_nltk)

    print('keywords', '#' * 60)
    print(keywords_list)

    return keywords_list


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.wikisearch', ignore_result=False, wp=WP(), es=ES('concepts'))
def wikisearch_task(self, keywords_list, fraction=(0, 1), method='es-base'):
    """
    Returns top 10 results for Wikipedia pages relevant to the keywords.

    Args:
        keywords_list (list(str)): List containing the keyword sets to search for among Wikipedia pages.
        fraction (tuple): Portion of the keywords_list to be processed, e.g. (1/3, 2/3) means only the middle third of
        the list is considered.
        method (str): Method to retrieve the wikipedia pages. It can be either "wikipedia-api", to use the
        Wikipedia API, or one of {"es-base", "es-score"}, to use elasticsearch, returning as score the inverse
        of the searchrank or the actual elasticsearch score, respectively. Default: 'es-base'. Fallback:
        'wikipedia-api'.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
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
                results = self.es.search(keywords)
            except Exception:
                print('[ERROR] Error connecting to elasticsearch cluster. Falling back to Wikipedia API.')
                results = self.wp.search(keywords)

            # Fallback to Wikipedia API if no results from elasticsearch
            if len(results) == 0:
                print('[WARNING] No results from elasticsearch cluster. Falling back to Wikipedia API.')
                results = self.wp.search(keywords)

        # Replace score with linear function on Searchrank if needed
        if method != 'es-score':
            if len(results) >= 2:
                results['SearchScore'] = 1 - (results['Searchrank'] - 1) / (len(results) - 1)
            else:
                results['SearchScore'] = 1

        # Ignore set of keywords if no pages are found
        if len(results) == 0:
            continue

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
    # Concatenate all results in a single DataFrame
    results = pd.concat(results, ignore_index=True)

    print('after wikisearch', '#' * 60)
    print(results)

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.compute_scores', ignore_result=False, ontology=ontology, graph=graph)
def compute_scores_task(self, results, ontology_score_smoothing=True, graph_score_smoothing=True, keywords_score_smoothing=True):
    if len(results) == 0:
        return results

    # Compute GraphScore
    results = self.graph.add_graph_score(results, smoothing=graph_score_smoothing)

    print('after graph', '#' * 60)
    print(results)

    # Compute OntologyLocalScore and OntologyGlobalScore
    self.ontology.graph = self.graph
    results = self.ontology.add_ontology_scores(results, smoothing=ontology_score_smoothing)

    print('after ontology', '#' * 60)
    print(results)

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

    print('after keywords', '#' * 60)
    print(results)

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text_10.aggregate_and_filter', ignore_result=False)
def aggregate_and_filter_task(self, results, coef=0.5, epsilon=0.1, min_votes=5):
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

    # Aggregate over pages, compute sum and max of scores
    results = results.groupby(by=['PageID', 'PageTitle']).aggregate(
        SearchScoreSum=('SearchScore', 'sum'),
        SearchScoreMax=('SearchScore', 'max'),
        LevenshteinScoreSum=('LevenshteinScore', 'sum'),
        LevenshteinScoreMax=('LevenshteinScore', 'max'),
        OntologyLocalScoreSum=('OntologyLocalScore', 'sum'),
        OntologyLocalScoreMax=('OntologyLocalScore', 'max'),
        OntologyGlobalScoreSum=('OntologyGlobalScore', 'sum'),
        OntologyGlobalScoreMax=('OntologyGlobalScore', 'max'),
        GraphScoreSum=('GraphScore', 'sum'),
        GraphScoreMax=('GraphScore', 'max'),
        KeywordsScoreSum=('KeywordsScore', 'sum'),
        KeywordsScoreMax=('KeywordsScore', 'max')
    ).reset_index()
    score_columns = ['SearchScore', 'LevenshteinScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'GraphScore',
                     'KeywordsScore']

    # Normalise sum scores to [0, 1]
    for column in score_columns:
        results[f'{column}Sum'] = results[f'{column}Sum'] / results[f'{column}Sum'].max()

    # Take convex combination of the two columns
    assert 0 <= coef <= 1, f'Normalisation coefficient {coef} is not in [0, 1]'
    for column in score_columns:
        results[column] = coef * results[f'{column}Sum'] + (1 - coef) * results[f'{column}Max']

    results = results[['PageID', 'PageTitle'] + score_columns]

    print('after aggregation', '#' * 60)
    print(results)

    ################################################################
    # Filtering of results                                         #
    ################################################################

    # Filter results with low scores through a majority vote among all scores
    # To be kept, we require a concept to have at least min_votes out of 6 scores to be significant (>= epsilon)
    assert 0 <= epsilon <= 1, f'Filtering threshold {epsilon} is not in [0, 1]'
    assert 0 <= min_votes <= 6, f'Filtering minimum number of votes {min_votes} is not in [0, 6]'
    votes = pd.DataFrame()
    for column in score_columns:
        votes[column] = (results[column] >= epsilon).astype(int)
    votes = votes.sum(axis=1)
    results = results[votes >= min_votes]

    ################################################################
    # Compute mixed score                                          #
    ################################################################

    # Compute mixed score as a convex combination of the different scores,
    # with prescribed coefficients found after running some analyses on manually tagged data.
    coefficients = pd.DataFrame({
        'SearchScore': [0.2],
        'LevenshteinScore': [0.15],
        'OntologyLocalScore': [0.15],
        'OntologyGlobalScore': [0.1],
        'GraphScore': [0.1],
        'KeywordsScore': [0.3]
    })
    results['MixedScore'] = results[score_columns] @ coefficients.transpose()

    print('after filter', '#' * 60)
    print(results)

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_10.sleeper', ignore_result=False)
def text_test_task(self):
    sleep(15)
    print('it worked')
    return 0


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_10.init', ignore_result=False, graph=graph, ontology=ontology)
def text_init_task(self):
    # This task initialises the text celery worker by loading into memory the graph and ontology tables

    print('Loading graph and ontology tables...')
    self.graph.fetch_from_db()
    self.ontology.fetch_from_db()
    print('Loaded')

    return True
