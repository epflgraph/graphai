import configparser
from elasticsearch import Elasticsearch

from concept_detection.text.utils import decode_url_title
from definitions import CONFIG_DIR

es_config = configparser.ConfigParser()
es_config.read(f'{CONFIG_DIR}/es.ini')
es = Elasticsearch([f'{es_config["ES"].get("host")}:{es_config["ES"].getint("port")}'])


def preprocess(keyword_list):
    """
    Cleans all the keywords in the given keyword list by applying the decode_url_title function.

    Args:
        keyword_list (list of str): List of keywords to be cleaned.

    Returns:
        list of str: List of cleaned keywords.
    """
    return [decode_url_title(keywords) for keywords in keyword_list]


def wikisearch(keyword_list, es_scores):
    # Clean all keywords in keyword_list
    keyword_list = preprocess(keyword_list)

    results = []
    for keywords in keyword_list:
        query = {
            'bool': {
                'must': {
                    'match': {'content': keywords}
                },
                'should': {
                    'match': {'title': keywords}
                }
            }
        }
        response = es.search(index='wikimath', query=query)
        hits = response['hits']['hits']
        max_score = response['hits']['max_score']

        pages = []
        for i in range(len(hits)):
            page_id = hits[i]['_source']['id']
            page_title = hits[i]['_source']['title']
            searchrank = i + 1
            if es_scores:
                score = (hits[i]['_score'] / max_score) if max_score else 0
            else:
                score = 1 / searchrank

            pages.append({
                'page_id': page_id,
                'page_title': page_title,
                'searchrank': searchrank,
                'score': score
            })

        results.append({
            'keywords': keywords,
            'pages': pages
        })

    return results
