from elasticsearch_interface.es import ESLex


RETRIEVAL_PARAMS = {
    "lex": {
        "default_index": "ramtin_lex_index",
        "retrieval_class": ESLex,
        "model": "all-MiniLM-L12-v2"
    }
}
