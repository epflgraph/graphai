from interfaces.es import ES
from models.wikisearch_result import WikisearchResult
from concept_detection.search.wikisearch import clean

es = ES('wikipages')


def wikisearch(keyword_list, es_scores):
    # Clean all keywords in keyword_list
    keyword_list = clean(keyword_list)

    results = []
    for keywords in keyword_list:
        pages = es.search(keywords)

        if not es_scores:
            # Replace score with 1/searchrank
            for page in pages:
                page.score = 1 / page.searchrank

        results.append(WikisearchResult(keywords=keywords, pages=pages))

    return results
