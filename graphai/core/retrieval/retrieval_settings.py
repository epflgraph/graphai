from elasticsearch_interface.es import ESLex
try:
    from elasticsearch_interface.es import ESServiceDesk
except ImportError:
    print('Outdated elasticsearch-interface version, reverting to default ES retriever class')
    ESServiceDesk = ESLex

RETRIEVAL_PARAMS = dict()
RETRIEVAL_PARAMS["lex"] = {
    "default_index": "ramtin_lex_index",
    "retrieval_class": ESLex,
    "model": "all-MiniLM-L12-v2"
}

RETRIEVAL_PARAMS["servicedesk"] = {
    "default_index": "ramtin_servicedesk_index",
    "retrieval_class": ESServiceDesk,
    "model": "all-MiniLM-L12-v2"
}
