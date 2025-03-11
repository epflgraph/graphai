from elasticsearch_interface.es import ESLex, ESServiceDesk


RETRIEVAL_PARAMS = {
    "lex": {
        "default_index": "ramtin_lex_index",
        "retrieval_class": ESLex,
        "model": "all-MiniLM-L12-v2"
    },
    "servicedesk": {
        "default_index": "ramtin_servicedesk_index",
        "retrieval_class": ESServiceDesk,
        "model": "all-MiniLM-L12-v2"
    }
}
