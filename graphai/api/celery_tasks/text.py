import pandas as pd

from celery import shared_task

import Levenshtein

from graphai.api.common.graph import graph
from graphai.api.common.ontology import ontology

from graphai.core.interfaces.wp import WP
from graphai.core.interfaces.es import ES

from graphai.core.text.keywords import get_keywords


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text.extract_keywords', ignore_result=False)
def extract_keywords_task(self, raw_text, use_nltk=False):
    return get_keywords(raw_text, use_nltk)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text.wikisearch', ignore_result=False, wp=WP(), es=ES('concepts'))
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
            results = self.es.search(keywords)

            # Fallback to Wikipedia API
            if len(results) == 0:
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
             name='text.wikisearch_callback', ignore_result=False)
def wikisearch_callback_task(self, results):
    # Concatenate all results in a single DataFrame
    results = pd.concat(results, ignore_index=True)

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text.compute_scores', ignore_result=False, ontology=ontology, graph=graph)
def compute_scores_task(self, results):
    if len(results) == 0:
        return results

    # Compute ontology score
    results = self.ontology.add_ontology_scores(results)

    # Compute graph score
    results = self.graph.add_graph_score(results)

    # Compute keywords score aggregating ontology global score over Keywords as an indicator for low-quality keywords
    results = pd.merge(
        results,
        results.groupby(by=['Keywords']).aggregate(KeywordsScore=('OntologyGlobalScore', 'sum')).reset_index(),
        how='left',
        on=['Keywords']
    )
    results['KeywordsScore'] = results['KeywordsScore'] / results['KeywordsScore'].max()

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text.aggregate_and_filter', ignore_result=False)
def aggregate_and_filter_task(self, results):
    if len(results) == 0:
        return results

    # Aggregate over pages
    results = results.groupby(by=['PageID', 'PageTitle']).aggregate(
        SearchScore=('SearchScore', 'sum'),
        LevenshteinScore=('LevenshteinScore', 'sum'),
        OntologyLocalScore=('OntologyLocalScore', 'sum'),
        OntologyGlobalScore=('OntologyGlobalScore', 'sum'),
        GraphScore=('GraphScore', 'sum'),
        KeywordsScore=('KeywordsScore', 'sum')
    ).reset_index()
    score_columns = ['SearchScore', 'LevenshteinScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'GraphScore',
                     'KeywordsScore']

    # Normalise scores to [0, 1]
    for column in score_columns:
        results[column] = results[column] / results[column].max()

    # Filter results with low scores through a majority vote among all scores
    # To be kept, we require a concept to have at least 5 out of 6 scores to be significant (>= epsilon)
    epsilon = 0.1
    votes = pd.DataFrame()
    for column in score_columns:
        votes[column] = (results[column] >= epsilon).astype(int)
    votes = votes.sum(axis=1)
    results = results[votes >= 5]
    results = results.sort_values(by='PageTitle')

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

    return results
