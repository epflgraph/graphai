import configparser
from elasticsearch import Elasticsearch

from definitions import CONFIG_DIR

from concept_detection.types.page_result import PageResult


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
        hits = search['hits']['hits']

        return [
            PageResult(
                page_id=hits[i]['_source']['id'],
                page_title=hits[i]['_source']['title'],
                searchrank=(i + 1),
                score=(1 / (i + 1))
            )
            for i in range(len(hits))
        ]

    def indices(self):
        return self.es.cat.indices(index=self.index, format='json', v=True)
