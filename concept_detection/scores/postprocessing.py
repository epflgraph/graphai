import numpy as np
import pandas as pd

import Levenshtein


def compute_scores(wikisearch_results, graph_results, page_id_titles, logger, debug=False):
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

    if debug:
        logger.info(f'wikisearch_results: {wikisearch_results}')

    # Convert wikisearch results to table
    wikisearch_table = []
    for wikisearch_result in wikisearch_results:
        keywords = wikisearch_result.keywords
        pages = wikisearch_result.pages
        for page in pages:
            wikisearch_table.append([keywords, page.page_id, page.page_title, page.searchrank, page.score])
    wikisearch_df = pd.DataFrame(wikisearch_table, columns=['keywords', 'page_id', 'page_title_0', 'searchrank', 'search_score'])

    if debug:
        logger.info(f'wikisearch_df: {wikisearch_df}')

    if debug:
        logger.info(f'graph_results: {graph_results}')

    # Convert graph scores to table
    graph_table = []
    for graph_result in graph_results:
        graph_table.append([graph_result['source_page_id'], graph_result['target_page_id'], graph_result['score']])
    graph_df = pd.DataFrame(graph_table, columns=['page_id', 'anchor_page_id', 'graph_score'])

    if debug:
        logger.info(f'graph_df: {graph_df}')

    # Merge graph and wikisearch dataframes
    merged_df = pd.merge(graph_df, wikisearch_df, how='left', on='page_id')

    # Keep only non-zero graph scores
    merged_df = merged_df[merged_df['graph_score'] > 0]

    # Sort values by keywords, searchrank and graph_score
    merged_df = merged_df.sort_values(by=['keywords', 'searchrank', 'graph_score'], ascending=[True, True, False])

    if debug:
        logger.info(f'merged_df: {merged_df}')

    # If empty, return
    if len(merged_df) == 0:
        return []

    # Calculate median graph scores
    median_scores_df = merged_df[['keywords', 'searchrank', 'page_id', 'page_title_0', 'graph_score', 'search_score']].groupby(
        ['keywords', 'searchrank', 'page_id', 'page_title_0']).median().rename(
        columns={'graph_score': 'median_graph_score'}).reset_index()

    if debug:
        logger.info(f'median_scores_df: {median_scores_df}')

    # Join page titles
    page_id_title_table = []
    for page_id in set(median_scores_df['page_id']):
        page_id_title_table.append([page_id, page_id_titles.get(str(page_id), '').replace('<squote/>', "'")])
    page_id_title_df = pd.DataFrame(page_id_title_table, columns=['page_id', 'page_name'])
    scores_df = pd.merge(median_scores_df, page_id_title_df, how='left', on='page_id')

    if debug:
        logger.info(f'scores_df (with page names): {scores_df}')

    # Calculate search-graph score ratio
    scores_df['search_graph_ratio'] = scores_df.apply(lambda r: r['search_score'] * r['median_graph_score'], axis=1)

    if debug:
        logger.info(f'scores_df (with sr_score and sr_graph_ratio): {scores_df}')

    # Filter rows with low search-graph ratio
    scores_df = scores_df[scores_df['search_graph_ratio'] >= 0.1]

    if debug:
        logger.info(f'scores_df (filtering search_graph_ratio >= 0.1): {scores_df}')

    # If empty, return
    if len(scores_df) == 0:
        return []

    # Calculate Levenshtein score
    scores_df['levenshtein_score'] = scores_df.apply(lambda r: Levenshtein.ratio(r['keywords'], r['page_name']), axis=1)

    # Calculate mixed score
    def f(x):
        return 1 / (1 + np.exp(-8 * (x - 1 / 2)))
    scores_df['mixed_score'] = scores_df.apply(lambda r: r['search_graph_ratio'] * f(r['levenshtein_score']), axis=1)

    if debug:
        logger.info(f'scores_df (with lev and mixed scores): {scores_df}')

    # Fix page id data type
    scores_df['page_id'] = scores_df['page_id'].astype(int)

    # Add page titles
    scores_df['page_title'] = scores_df.apply(lambda r: r['page_name'].replace(' ', '_').capitalize(), axis=1)

    if debug:
        logger.info(f'scores_df (with page titles): {scores_df}')

    # Generate output dataframe
    scores_df = scores_df[
        ['keywords', 'page_id', 'page_title_0', 'page_title', 'searchrank', 'median_graph_score', 'search_graph_ratio',
         'levenshtein_score', 'mixed_score']]

    if debug:
        logger.info(f'scores_df (output form): {scores_df}')

    # Return it as a dictionary
    return scores_df.to_dict(orient='records')
