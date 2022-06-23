import numpy as np
import pandas as pd

import Levenshtein


def log(msg, logger, debug):
    if debug:
        logger.info(msg)


def get_wikisearch_df(wikisearch_results):
    wikisearch_table = []
    for wikisearch_result in wikisearch_results:
        keywords = wikisearch_result.keywords
        pages = wikisearch_result.pages
        for page in pages:
            wikisearch_table.append([keywords, page.page_id, page.page_title, page.searchrank, page.score])

    columns = ['keywords', 'page_id', 'page_title', 'searchrank', 'search_score']
    return pd.DataFrame(wikisearch_table, columns=columns)


def get_graph_df(graph_results):
    graph_table = []
    for graph_result in graph_results:
        graph_table.append([graph_result['source_page_id'], graph_result['target_page_id'], graph_result['score']])

    columns = ['page_id', 'anchor_page_id', 'graph_score']
    return pd.DataFrame(graph_table, columns=columns)


def compute_scores(wikisearch_results, graph_results, logger, debug=False):
    """
    Takes the results of a wikisearch and a graph search and creates a dataframe with them as well as new derived scores.

    Args:
        wikisearch_results (list): The results of a wikisearch. Each element is a dictionary containing the keywords and
            a list of Wikipedia pages for those keywords.
        graph_results (list): The results of a graph search. Each element is a dictionary containing the source and the
            anchor page ids and their graph score.
        page_id_titles (dict): The mapping from page ids to their titles.
        logger (Logger): A logger to use for debugging purposes.
        debug (bool): Whether the function prints the results at each step. Default: False.

    Returns:
        dict: A dictionary representing a pandas dataframe with all the scores.
    """

    # Convert wikisearch results to DataFrame
    log(f'wikisearch_results: {wikisearch_results}', logger, debug)
    wikisearch_df = get_wikisearch_df(wikisearch_results)
    log(f'wikisearch_df: {wikisearch_df}', logger, debug)

    # Convert graph scores to DataFrame and keep only non-zero scores
    log(f'graph_results: {graph_results}', logger, debug)
    graph_df = get_graph_df(graph_results)
    graph_df = graph_df[graph_df['graph_score'] > 0]
    log(f'graph_df: {graph_df}', logger, debug)

    # Merge graph and wikisearch dataframes, and sort values by keywords, searchrank and graph_score
    merged_df = pd.merge(graph_df, wikisearch_df, how='left', on='page_id')
    # merged_df = merged_df.sort_values(by=['keywords', 'searchrank', 'graph_score'], ascending=[True, True, False])
    log(f'merged_df: {merged_df}', logger, debug)

    # If dataframe is empty, return
    if len(merged_df) == 0:
        return []

    # Calculate median graph scores
    select_columns = ['keywords', 'searchrank', 'page_id', 'page_title', 'graph_score', 'search_score']
    group_columns = ['keywords', 'searchrank', 'page_id', 'page_title']
    scores_df = merged_df[select_columns].groupby(group_columns, as_index=False).median()
    scores_df = scores_df.rename(columns={'graph_score': 'median_graph_score'})
    log(f'scores_df (after aggregation with median scores): {scores_df}', logger, debug)

    # Calculate search-graph score ratio
    scores_df['search_graph_ratio'] = scores_df.apply(lambda r: r['search_score'] * r['median_graph_score'], axis=1)
    log(f'scores_df (with sr_score and sr_graph_ratio): {scores_df}', logger, debug)

    # Filter rows with low search-graph ratio
    scores_df = scores_df[scores_df['search_graph_ratio'] >= 0.1]
    log(f'scores_df (filtering search_graph_ratio >= 0.1): {scores_df}', logger, debug)

    # If dataframe is empty, return
    if len(scores_df) == 0:
        return []

    # Calculate Levenshtein score
    scores_df['levenshtein_score'] = scores_df.apply(lambda r: Levenshtein.ratio(r['keywords'], r['page_title'].lower()), axis=1)

    # Calculate mixed score
    def f(x):
        return 1 / (1 + np.exp(-8 * (x - 1 / 2)))   # f pulls values in [0, 1] away from 1/2, exaggerating differences

    scores_df['mixed_score'] = scores_df['search_graph_ratio'] * f(scores_df['levenshtein_score'])
    log(f'scores_df (with lev and mixed scores): {scores_df}', logger, debug)

    # Fix page id data type
    scores_df['page_id'] = scores_df['page_id'].astype(int)

    # Replace ' ' with '_' in titles to make it consistent with db
    scores_df['page_title'] = scores_df.apply(lambda r: r['page_title'].replace(' ', '_'), axis=1)
    log(f'scores_df (with page titles): {scores_df}', logger, debug)

    # Generate output dataframe
    output_columns = ['keywords', 'page_id', 'page_title', 'searchrank',
                      'median_graph_score', 'search_graph_ratio', 'levenshtein_score', 'mixed_score']
    scores_df = scores_df[output_columns]
    log(f'scores_df (output form): {scores_df}', logger, debug)

    # Return it as a dictionary
    return scores_df.to_dict(orient='records')
