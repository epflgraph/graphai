import configparser
import math

from elasticsearch import Elasticsearch

from definitions import CONFIG_DIR

from concept_detection.types.page_result import PageResult


def es_bool(must=None, must_not=None, should=None):
    query = {
        'bool': {}
    }

    if must:
        query['bool']['must'] = must

    if must_not:
        query['bool']['must_not'] = must_not

    if should:
        query['bool']['should'] = should

    return query


def es_match(field, text, boost=1):
    return {
        'match': {
            field: {
                'query': text,
                'boost': boost
            }
        }
    }


def es_exp_decay_function(field, scale):
    # Decay function is exp(-x/scale)
    # How to pick a value for scale:
    # Page score is multiplied by a factor
    #   * For pages with field < scale, the factor will be > 1/e (~= 0.37)
    #   * For pages with field > scale, the factor will be < 1/e (~= 0.37)
    #
    # Pages with high field values (>scale) will have their score roughly thirded, or worse.
    return {
        'exp': {
            field: {
                'origin': 0,
                'offset': 0,
                'scale': scale,
                'decay': 1/math.e
            }
        }
    }


def es_function_score(query, functions, boost_mode='multiply'):
    return {
        'function_score': {
            'query': query,
            'functions': functions,
            'boost_mode': boost_mode
        }
    }


def es_exp_booster_source(scale=1e9, max_boost=0.5):
    return {
        'source': f"_score * (1 + {max_boost} * Math.exp(- doc['id'].value/{scale}))"
    }


def es_script_score(query, script):
    return {
        'script_score': {
            'query': query,
            'script': script
        }
    }


class ES:
    def __init__(self):
        self.es_config = configparser.ConfigParser()
        self.es_config.read(f'{CONFIG_DIR}/es.ini')
        self.host = self.es_config["ES"].get("host")
        self.port = self.es_config["ES"].get("port")
        self.index = self.es_config["ES"].get("index")

        self.es = Elasticsearch([f'{self.host}:{self.port}'])

    def _search(self, query, limit=10):
        return self.es.search(index=self.index, query=query, size=limit)

    def _results_from_search(self, search):
        hits = search['hits']['hits']

        return [
            PageResult(
                page_id=hits[i]['_source']['id'],
                page_title=hits[i]['_source']['title'],
                searchrank=(i + 1),
                score=hits[i]['_score']
            )
            for i in range(len(hits))
        ]

    def search(self, text, limit=10):
        query = es_bool(
            must=es_match('content', text),
            should=es_match('title', text)
        )
        search = self._search(query, limit=limit)
        return self._results_from_search(search)

    def search_boost_title(self, text, limit=10, boost=2):
        query = es_bool(
            must=es_match('content', text),
            should=es_match('title', text, boost=boost)
        )
        search = self._search(query, limit=limit)
        return self._results_from_search(search)

    def search_penalize_title(self, text, limit=10, penalty=2):
        query = es_bool(
            must=es_match('content', text),
            should=es_bool(
                must_not=es_match('title', text, boost=penalty)
            )
        )
        search = self._search(query, limit=limit)
        return self._results_from_search(search)

    def search_decay_page_id(self, text, limit=10, scale=1e9):
        query = es_function_score(
            query=es_bool(
                must=es_match('content', text),
                should=es_match('title', text)
            ),
            functions=[es_exp_decay_function(field='id', scale=scale)]
        )
        search = self._search(query, limit=limit)
        return self._results_from_search(search)

    def search_boost_low_page_id(self, text, limit=10, scale=1e9, max_boost=0.5):
        query = es_script_score(
            query=es_bool(
                must=es_match('content', text),
                should=es_match('title', text)
            ),
            script=es_exp_booster_source(scale=scale, max_boost=max_boost)
        )
        search = self._search(query, limit=limit)
        return self._results_from_search(search)

    def indices(self):
        return self.es.cat.indices(index=self.index, format='json', v=True)

