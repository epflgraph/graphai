import wikipedia
import ray

from concept_detection.text.utils import decode_url_title

# Set wikipedia language
wikipedia.set_lang('en')

# Init ray
ray.init(namespace="wikisearch", include_dashboard=False, log_to_driver=True)


@ray.remote
class WikisearchActor:
    """
    Class representing a ray Actor to perform wikisearch in parallel.
    """
    def wikisearch(self, keywords):
        """
        Searches Wikipedia API for input keywords. Returns top 10 results.

        Args:
            keywords (str): Text to input in Wikipedia's search field.

        Returns:
            dict(
                'keywords': keywords (str),
                'pages': list of dict
            )
        """
        # Check length of keywords. Return if too long
        if len(keywords) > 92:
            print('Keywords exceed maximum length of 92 characters.')
            return {
                'keywords': keywords,
                'pages': {}
            }

        # Run search on Wikipedia API
        top_page_titles = wikipedia.search(keywords)

        # Create pages object with escaped title
        n = min(len(top_page_titles), 10)
        pages = [
            {
                'page_id': 0,
                'page_title': top_page_titles[i].lower().replace('_', ' ').replace("'", '<squote/>').replace('"', '<dquote/>'),
                'searchrank': i + 1,
                'score': 1 / (i + 1)
            }
            for i in range(n)
        ]

        # Return dictionary with keywords and pages
        return {
            'keywords': keywords,
            'pages': pages
        }


# Instantiate ray actor list
n_actors = 16
actors = [WikisearchActor.remote() for i in range(n_actors)]


def preprocess(keyword_list):
    """
    Cleans all the keywords in the given keyword list by applying the decode_url_title function.

    Args:
        keyword_list (list of str): List of keywords to be cleaned.

    Returns:
        list of str: List of cleaned keywords.
    """
    return [decode_url_title(keywords) for keywords in keyword_list]


def postprocess(results, page_title_ids):
    """
    Modifies the given results to include page ids in addition to page titles.

    Args:
        results (list of dict): List of wikisearch results.
        page_title_ids (dict of str: int): Mapping from page titles to ids.

    Returns:
        list of dict: The list of wikisearch results, with updated page ids based on the page titles.
    """

    for result in results:
        for i in range(len(result['pages'])):
            # Get page id for the given page title from mapping if present, None otherwise
            page_title = result['pages'][i]['page_title']
            result['pages'][i]['page_id'] = page_title_ids.get(page_title, None)

    return results


def extract_page_ids(results):
    """
    Iterates through the given wikisearch results and returns a list with all the page ids.

    Args:
        results (list of dict): List of wikisearch results.

    Returns:
        list of int: List of all page ids present along all results.
    """
    return list(set(page['page_id'] for result in results for page in result['pages']))


def extract_anchor_page_ids(results, max_n=3):
    """
    Iterates through the given wikisearch results and returns a list with the most relevant page ids.

    Args:
        results (list of dict): List of wikisearch results.
        max_n (int): Maximum number of page ids returned. Default: 3.

    Returns:
        list of int: List of the most relevant page ids present along all results.
    """

    page_scores = {}
    for result in results:
        for page in result['pages']:
            page_id = page['page_id']

            if page_id is None:
                continue

            if page_id in page_scores:
                page_scores[page_id] += page['page_scores']
            else:
                page_scores[page_id] = page['page_scores']

    high_scores = sorted(page_scores.values(), reverse=True)[:max_n]

    return list(set(page_id for page_id in page_scores if page_scores[page_id] in high_scores))


def run_wikisearch(keywords):
    """
    Searches Wikipedia API for input keywords. Returns top 10 results.

    Args:
        keywords (str): Text to input in Wikipedia's search field.

    Returns:
        dict: Dictionary with keys 'keywords' (str) and 'pages' (list of dict)
    """

    # Check length of keywords. Return if too long
    if len(keywords) > 92:
        print('Keywords exceed maximum length of 92 characters.')
        return {
            'keywords': keywords,
            'pages': {}
        }

    # Run search on Wikipedia API
    top_page_titles = wikipedia.search(keywords)

    # Create pages object with escaped title
    n = min(len(top_page_titles), 10)
    pages = [
        {
            'page_id': 0,
            'page_title': top_page_titles[i].lower().replace('_', ' ').replace("'", '<squote/>').replace('"', '<dquote/>'),
            'searchrank': i + 1,
            'score': 1 / (i + 1)
        }
        for i in range(n)
    ]

    # Return dictionary with keywords and pages
    return {
        'keywords': keywords,
        'pages': pages
    }


def wikisearch(keyword_list, page_title_ids, how='ray'):
    """
    Searches Wikipedia API for all keywords in the list, sequentially or in parallel.

    Args:
        keyword_list (list of str): List of keywords to perform the wikisearch.
        page_title_ids (dict of str: int): Mapping from page titles to ids.
        how (str): String specifying whether to run the wikisearch sequentially or in parallel.
            Possible values: 'seq' (sequentially) or 'ray' (parallel). Default: 'ray'.

    Returns:
        list of dict: List of wikisearch results.
    """

    # Clean all keywords in keyword_list
    keyword_list = preprocess(keyword_list)

    results = []
    if how == 'seq' or how == 'all':
        for keywords in keyword_list:
            results.append(run_wikisearch(keywords))

    if how == 'ray' or how == 'all':
        # Execute wikisearch in parallel
        results = [actors[i % n_actors].wikisearch.remote(keyword_list[i]) for i in range(len(keyword_list))]

        # Wait for the results
        results = ray.get(results)

    # Modify results to include page ids instead of titles
    results = postprocess(results, page_title_ids)

    return results
