import configparser
from elasticsearch import Elasticsearch

from definitions import CONFIG_DIR


class ES:
    def __init__(self):
        self.es_config = configparser.ConfigParser()
        self.es_config.read(f'{CONFIG_DIR}/es.ini')
        self.host = self.es_config["ES"].get("host")
        self.port = self.es_config["ES"].get("port")
        self.index = self.es_config["ES"].get("index")

        self.es = Elasticsearch([f'{self.host}:{self.port}'])

    def search(self, query):
        return self.es.search(index=self.index, query=query)

    def minsearch(self, query):
        search = self.search(query)

        return [{
            'page_id': hit['_source']['id'],
            'page_title': hit['_source']['title'],
            'score': hit['_score'],
        } for hit in search['hits']['hits']]

    def indices(self):
        return self.es.cat.indices(index=self.index, format='json', v=True)
