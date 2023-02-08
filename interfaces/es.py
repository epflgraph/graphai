import configparser

from ssl import create_default_context

import pandas as pd
from elasticsearch import Elasticsearch

from definitions import CONFIG_DIR


def es_bool(must=None, must_not=None, should=None, filter=None):
    """
    Build elasticsearch bool clause with given arguments.

    Returns:
        dict
    """

    query = {
        'bool': {}
    }

    if must is not None:
        query['bool']['must'] = must

    if must_not is not None:
        query['bool']['must_not'] = must_not

    if should is not None:
        query['bool']['should'] = should

    if filter is not None:
        query['bool']['filter'] = filter

    return query


def es_match(field, text, boost=None, operator=None):
    """
    Build elasticsearch match clause with given arguments.

    Returns:
        dict
    """

    query = {
        'match': {
            field: {
                'query': text
            }
        }
    }

    if boost is not None:
        query['match'][field]['boost'] = boost

    if operator is not None:
        query['match'][field]['operator'] = operator

    return query


def es_multi_match(fields, text, type=None, boost=None, minimum_should_match=None, operator=None):
    """
    Build elasticsearch multi_match clause with given arguments.

    Returns:
        dict
    """

    query = {
        'multi_match': {
            'fields': fields,
            'query': text
        }
    }

    if type is not None:
        query['multi_match']['type'] = type

    if boost is not None:
        query['multi_match']['boost'] = boost

    if minimum_should_match is not None:
        query['multi_match']['minimum_should_match'] = minimum_should_match

    if operator is not None:
        query['multi_match']['operator'] = operator

    return query


def es_dis_max(queries):
    """
    Build elasticsearch dis_max clause with given arguments.

    Returns:
        dict
    """

    query = {
        'dis_max': {
            'queries': queries
        }
    }

    return query


class ES:
    """
    Base class to communicate with the elasticsearch EPFLGraph graphai instance.
    """

    def __init__(self, index):
        self.index = index

        self.es_config = configparser.ConfigParser()
        self.es_config.read(f'{CONFIG_DIR}/es.ini')
        self.host = self.es_config["ES"].get("host")
        self.port = self.es_config["ES"].get("port")
        self.username = self.es_config["ES"].get("username")
        self.cafile = self.es_config["ES"].get("cafile")
        password = self.es_config["ES"].get("password")

        context = create_default_context(cafile=self.cafile)
        self.es = Elasticsearch([f'https://{self.username}:{password}@{self.host}:{self.port}'], ssl_context=context, timeout=3600)

    def _search(self, query, limit=10, source=None, explain=False, rescore=None):
        return self.es.search(index=self.index, query=query, source=source, rescore=rescore, size=limit, explain=explain, profile=True)

    @staticmethod
    def _to_dataframe(search):
        hits = search['hits']['hits']

        table = [[hits[i]['_source']['id'], hits[i]['_source']['title'], (i + 1), hits[i]['_score']] for i in range(len(hits))]

        return pd.DataFrame(table, columns=['PageID', 'PageTitle', 'Searchrank', 'SearchScore'])

    def search(self, text, limit=10):
        """
        Perform elasticsearch search query.

        Args:
            text (str): Query text for the search.
            limit (int): Maximum number of returned results.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            unique in 'PageID', with the wikisearch results the given keywords set.
        """

        return self.search_mediawiki(text, limit=limit)

    def search_mediawiki(self, text, limit=10):
        """
        Perform elasticsearch search query using the mediawiki query structure.

        Args:
            text (str): Query text for the search.
            limit (int): Maximum number of returned results.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            unique in 'PageID', with the wikisearch results the given keywords set.
        """

        query = es_bool(
            should=[
                es_multi_match(fields=['all_near_match^10', 'all_near_match_asciifolding^7.5'], text=text),
                es_bool(
                    filter=[
                        es_bool(
                            should=[
                                es_match('all', text=text, operator='and'),
                                es_match('all.plain', text=text, operator='and')
                            ]
                        )
                    ],
                    should=[
                        es_multi_match(fields=['title^3', 'title.plain^1'], text=text, type='most_fields', boost=0.3, minimum_should_match=1),
                        es_multi_match(fields=['category^3', 'category.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                        es_multi_match(fields=['heading^3', 'heading.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                        es_multi_match(fields=['auxiliary_text^3', 'auxiliary_text.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                        es_multi_match(fields=['file_text^3', 'file_text.plain^1'], text=text, type='most_fields', boost=0.5, minimum_should_match=1),
                        es_dis_max([
                            es_multi_match(fields=['redirect^3', 'redirect.plain^1'], text=text, type='most_fields', boost=0.27, minimum_should_match=1),
                            es_multi_match(fields=['suggest'], text=text, type='most_fields', boost=0.2, minimum_should_match=1)
                        ]),
                        es_dis_max([
                            es_multi_match(fields=['text^3', 'text.plain^1'], text=text, type='most_fields', boost=0.6, minimum_should_match=1),
                            es_multi_match(fields=['opening_text^3', 'opening_text.plain^1'], text=text, type='most_fields', boost=0.5, minimum_should_match=1)
                        ]),
                    ]
                )
            ]
        )

        rescore = [
            {
                "window_size": 8192,
                "query": {
                    "query_weight": 1,
                    "rescore_query_weight": 1,
                    "score_mode": "total",
                    "rescore_query": {
                        "function_score": {
                            "score_mode": "sum",
                            "boost_mode": "sum",
                            "functions": [
                                {
                                    "script_score": {
                                        "script": {
                                            "source": "pow(doc['popularity_score'].value , 0.8) / ( pow(doc['popularity_score'].value, 0.8) + pow(8.0E-6,0.8))",
                                            "lang": "expression"
                                        }
                                    },
                                    "weight": 3
                                },
                                {
                                    "script_score": {
                                        "script": {
                                            "source": "pow(doc['incoming_links'].value , 0.7) / ( pow(doc['incoming_links'].value, 0.7) + pow(30,0.7))",
                                            "lang": "expression"
                                        }
                                    },
                                    "weight": 10
                                }
                            ]
                        }
                    }
                }
            },
            {
                "window_size": 448,
                "query": {
                    "query_weight": 1,
                    "rescore_query_weight": 10000,
                    "score_mode": "total",
                    "rescore_query": {
                        "bool": {
                            "should": [
                                {
                                    "constant_score": {
                                        "filter": {
                                            "match_all": {

                                            }
                                        },
                                        "boost": 100000
                                    }
                                },
                                {
                                    "sltr": {
                                        "model": "enwiki-20220421-20180215-query_explorer",
                                        "params": {
                                            "query_string": text
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        ]

        search = self._search(query, limit=limit, rescore=rescore)
        return ES._to_dataframe(search)

    def search_mediawiki_no_rescore(self, text, limit=10):
        """
        Perform elasticsearch search query using the mediawiki query structure, skipping the rescore part.

        Args:
            text (str): Query text for the search.
            limit (int): Maximum number of returned results.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            unique in 'PageID', with the wikisearch results the given keywords set.
        """

        query = es_bool(
            should=[
                es_multi_match(fields=['all_near_match^10', 'all_near_match_asciifolding^7.5'], text=text),
                es_bool(
                    filter=[
                        es_bool(
                            should=[
                                es_match('all', text=text, operator='and'),
                                es_match('all.plain', text=text, operator='and')
                            ]
                        )
                    ],
                    should=[
                        es_multi_match(fields=['title^3', 'title.plain^1'], text=text, type='most_fields', boost=0.3, minimum_should_match=1),
                        es_multi_match(fields=['category^3', 'category.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                        es_multi_match(fields=['heading^3', 'heading.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                        es_multi_match(fields=['auxiliary_text^3', 'auxiliary_text.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                        es_multi_match(fields=['file_text^3', 'file_text.plain^1'], text=text, type='most_fields', boost=0.5, minimum_should_match=1),
                        es_dis_max([
                            es_multi_match(fields=['redirect^3', 'redirect.plain^1'], text=text, type='most_fields', boost=0.27, minimum_should_match=1),
                            es_multi_match(fields=['suggest'], text=text, type='most_fields', boost=0.2, minimum_should_match=1)
                        ]),
                        es_dis_max([
                            es_multi_match(fields=['text^3', 'text.plain^1'], text=text, type='most_fields', boost=0.6, minimum_should_match=1),
                            es_multi_match(fields=['opening_text^3', 'opening_text.plain^1'], text=text, type='most_fields', boost=0.5, minimum_should_match=1)
                        ]),
                    ]
                )
            ]
        )

        search = self._search(query, limit=limit)
        return ES._to_dataframe(search)

    def search_mediawiki_no_plain(self, text, limit=10):
        """
        Perform elasticsearch search query using the mediawiki query structure, restricted to non-plain fields.

        Args:
            text (str): Query text for the search.
            limit (int): Maximum number of returned results.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            unique in 'PageID', with the wikisearch results the given keywords set.
        """

        query = es_bool(
            should=[
                es_match(field='all_near_match', text=text, boost=10),
                es_bool(
                    filter=[
                        es_match('all', text=text, operator='and')
                    ],
                    should=[
                        es_match(field='title', text=text, boost=0.9),
                        es_match(field='category', text=text, boost=0.15),
                        es_match(field='heading', text=text, boost=0.15),
                        es_match(field='auxiliary_text', text=text, boost=0.15),
                        es_match(field='file_text', text=text, boost=1.5),
                        es_dis_max([
                            es_match(field='redirect', text=text, boost=0.81),
                            es_match(field='suggest', text=text, boost=0.2)
                        ]),
                        es_dis_max([
                            es_match(field='text', text=text, boost=1.8),
                            es_match(field='opening_text', text=text, boost=1.5)
                        ])
                    ]
                )
            ]
        )

        rescore = [
            {
                "window_size": 8192,
                "query": {
                    "query_weight": 1,
                    "rescore_query_weight": 1,
                    "score_mode": "total",
                    "rescore_query": {
                        "function_score": {
                            "score_mode": "sum",
                            "boost_mode": "sum",
                            "functions": [
                                {
                                    "script_score": {
                                        "script": {
                                            "source": "pow(doc['popularity_score'].value , 0.8) / ( pow(doc['popularity_score'].value, 0.8) + pow(8.0E-6,0.8))",
                                            "lang": "expression"
                                        }
                                    },
                                    "weight": 3
                                },
                                {
                                    "script_score": {
                                        "script": {
                                            "source": "pow(doc['incoming_links'].value , 0.7) / ( pow(doc['incoming_links'].value, 0.7) + pow(30,0.7))",
                                            "lang": "expression"
                                        }
                                    },
                                    "weight": 10
                                }
                            ]
                        }
                    }
                }
            },
            {
                "window_size": 448,
                "query": {
                    "query_weight": 1,
                    "rescore_query_weight": 10000,
                    "score_mode": "total",
                    "rescore_query": {
                        "bool": {
                            "should": [
                                {
                                    "constant_score": {
                                        "filter": {
                                            "match_all": {

                                            }
                                        },
                                        "boost": 100000
                                    }
                                },
                                {
                                    "sltr": {
                                        "model": "enwiki-20220421-20180215-query_explorer",
                                        "params": {
                                            "query_string": text
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        ]

        search = self._search(query, limit=limit, rescore=rescore)
        return ES._to_dataframe(search)

    def search_mediawiki_restrict_4(self, text, limit=10):
        """
        Perform elasticsearch search query using the mediawiki query structure, restricted to the following fields:
        title, text, heading, opening_text

        Args:
            text (str): Query text for the search.
            limit (int): Maximum number of returned results.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            unique in 'PageID', with the wikisearch results the given keywords set.
        """

        query = es_bool(
            should=[
                es_bool(
                    filter=[
                        es_bool(
                            should=[
                                es_match('title', text=text, operator='and'),
                                es_match('title.plain', text=text, operator='and'),
                                es_match('text', text=text, operator='and'),
                                es_match('text.plain', text=text, operator='and'),
                                es_match('heading', text=text, operator='and'),
                                es_match('heading.plain', text=text, operator='and')
                            ]
                        )
                    ],
                    should=[
                        es_multi_match(fields=['title^3', 'title.plain^1'], text=text, type='most_fields', boost=0.3, minimum_should_match=1),
                        es_multi_match(fields=['heading^3', 'heading.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                        es_dis_max([
                            es_multi_match(fields=['text^3', 'text.plain^1'], text=text, type='most_fields', boost=0.6, minimum_should_match=1),
                            es_multi_match(fields=['opening_text^3', 'opening_text.plain^1'], text=text, type='most_fields', boost=0.5, minimum_should_match=1)
                        ])
                    ]
                )
            ]
        )

        rescore = [
            {
                "window_size": 8192,
                "query": {
                    "query_weight": 1,
                    "rescore_query_weight": 1,
                    "score_mode": "total",
                    "rescore_query": {
                        "function_score": {
                            "score_mode": "sum",
                            "boost_mode": "sum",
                            "functions": [
                                {
                                    "script_score": {
                                        "script": {
                                            "source": "pow(doc['popularity_score'].value , 0.8) / ( pow(doc['popularity_score'].value, 0.8) + pow(8.0E-6,0.8))",
                                            "lang": "expression"
                                        }
                                    },
                                    "weight": 3
                                },
                                {
                                    "script_score": {
                                        "script": {
                                            "source": "pow(doc['incoming_links'].value , 0.7) / ( pow(doc['incoming_links'].value, 0.7) + pow(30,0.7))",
                                            "lang": "expression"
                                        }
                                    },
                                    "weight": 10
                                }
                            ]
                        }
                    }
                }
            },
            {
                "window_size": 448,
                "query": {
                    "query_weight": 1,
                    "rescore_query_weight": 10000,
                    "score_mode": "total",
                    "rescore_query": {
                        "bool": {
                            "should": [
                                {
                                    "constant_score": {
                                        "filter": {
                                            "match_all": {

                                            }
                                        },
                                        "boost": 100000
                                    }
                                },
                                {
                                    "sltr": {
                                        "model": "enwiki-20220421-20180215-query_explorer",
                                        "params": {
                                            "query_string": text
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        ]

        search = self._search(query, limit=limit, rescore=rescore)
        return ES._to_dataframe(search)

    def search_mediawiki_restrict_2(self, text, limit=10):
        """
        Perform elasticsearch search query using the mediawiki query structure, restricted to the following fields:
        title, text

        Args:
            text (str): Query text for the search.
            limit (int): Maximum number of returned results.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            unique in 'PageID', with the wikisearch results the given keywords set.
        """

        query = es_bool(
            should=[
                es_bool(
                    filter=[
                        es_bool(
                            should=[
                                es_match('title', text=text, operator='and'),
                                es_match('title.plain', text=text, operator='and'),
                                es_match('text', text=text, operator='and'),
                                es_match('text.plain', text=text, operator='and')
                            ]
                        )
                    ],
                    should=[
                        es_multi_match(fields=['title^3', 'title.plain^1'], text=text, type='most_fields', boost=0.3, minimum_should_match=1),
                        es_multi_match(fields=['text^3', 'text.plain^1'], text=text, type='most_fields', boost=0.6, minimum_should_match=1)
                    ]
                )
            ]
        )

        rescore = [
            {
                "window_size": 8192,
                "query": {
                    "query_weight": 1,
                    "rescore_query_weight": 1,
                    "score_mode": "total",
                    "rescore_query": {
                        "function_score": {
                            "score_mode": "sum",
                            "boost_mode": "sum",
                            "functions": [
                                {
                                    "script_score": {
                                        "script": {
                                            "source": "pow(doc['popularity_score'].value , 0.8) / ( pow(doc['popularity_score'].value, 0.8) + pow(8.0E-6,0.8))",
                                            "lang": "expression"
                                        }
                                    },
                                    "weight": 3
                                },
                                {
                                    "script_score": {
                                        "script": {
                                            "source": "pow(doc['incoming_links'].value , 0.7) / ( pow(doc['incoming_links'].value, 0.7) + pow(30,0.7))",
                                            "lang": "expression"
                                        }
                                    },
                                    "weight": 10
                                }
                            ]
                        }
                    }
                }
            },
            {
                "window_size": 448,
                "query": {
                    "query_weight": 1,
                    "rescore_query_weight": 10000,
                    "score_mode": "total",
                    "rescore_query": {
                        "bool": {
                            "should": [
                                {
                                    "constant_score": {
                                        "filter": {
                                            "match_all": {

                                            }
                                        },
                                        "boost": 100000
                                    }
                                },
                                {
                                    "sltr": {
                                        "model": "enwiki-20220421-20180215-query_explorer",
                                        "params": {
                                            "query_string": text
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        ]

        search = self._search(query, limit=limit, rescore=rescore)
        return ES._to_dataframe(search)

    def indices(self):
        """
        Retrieve information about all elasticsearch indices.

        Returns:
            dict: elasticsearch response
        """
        return self.es.cat.indices(index=self.index, format='json', v=True)

    def index_doc(self, doc):
        """
        Index the given document.

        Args:
            doc (dict): Document to index.

        Returns:
            dict: elasticsearch response
        """

        if 'id' in doc:
            self.es.index(index=self.index, document=doc, id=doc['id'])
        else:
            self.es.index(index=self.index, document=doc)

    def create_index(self, settings=None, mapping=None):
        """
        Create index with the given settings and mapping.

        Args:
            settings (dict): Dictionary with elasticsearch settings, in that format.
            mapping (dict): Dictionary with elasticsearch mapping, in that format.

        Returns:
            dict: elasticsearch response
        """

        body = {}

        if settings is not None:
            body['settings'] = settings

        if mapping is not None:
            body['mappings'] = mapping

        if body:
            self.es.indices.create(index=self.index, body=body)
        else:
            self.es.indices.create(index=self.index)

    def delete_index(self):
        """
        Delete index.

        Returns:
            dict: elasticsearch response
        """

        self.es.indices.delete(index=self.index, ignore_unavailable=True)

    def refresh(self):
        """
        Refresh index.

        Returns:
            dict: elasticsearch response
        """

        self.es.indices.refresh(index=self.index)
