import configparser
from elasticsearch import Elasticsearch

from concept_detection.search.types import *
from concept_detection.search.wikisearch import clean
from definitions import CONFIG_DIR

es_config = configparser.ConfigParser()
es_config.read(f'{CONFIG_DIR}/es.ini')
es = Elasticsearch([f'{es_config["ES"].get("host")}:{es_config["ES"].getint("port")}'])


def wikisearch(keyword_list, es_scores):
    # Clean all keywords in keyword_list
    keyword_list = clean(keyword_list)

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

            pages.append(PageResult(
                page_id=page_id,
                page_title=page_title,
                searchrank=searchrank,
                score=score
            ))

        results.append(WikisearchResult(keywords=keywords, pages=pages))

    return results
