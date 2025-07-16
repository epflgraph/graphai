from elasticsearch_interface.es import (
    ESLex,
    ESServiceDesk,
    ESGeneralRAG
)

RETRIEVAL_PARAMS = dict()
RETRIEVAL_PARAMS["lex"] = {
    "default_index": "rag_lex_index",
    "retrieval_class": ESLex,
    "model": "all-MiniLM-L12-v2",
    "filters": ["lang"]
}

RETRIEVAL_PARAMS["servicedesk"] = {
    "default_index": "rag_servicedesk_index",
    "retrieval_class": ESServiceDesk,
    "model": "all-MiniLM-L12-v2",
    "filters": ["lang", "category"]
}

RETRIEVAL_PARAMS["sac"] = {
    "default_index": "rag_sac_index",
    "retrieval_class": ESLex,
    "model": "all-MiniLM-L12-v2",
    "filters": ["lang"]
}

RETRIEVAL_PARAMS["default"] = {
    "default_index": "rag_%s_index",
    "retrieval_class": ESGeneralRAG,
    "model": "all-MiniLM-L12-v2",
    "filters": None
}
