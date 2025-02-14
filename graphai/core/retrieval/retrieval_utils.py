from graphai.core.common.config import config
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS


def search_lex(text, embedding=None, lang=None, limit=10):
    try:
        lex_retriever = RETRIEVAL_PARAMS['lex']['retrieval_class'](
            config['elasticsearch'],
            index=config['elasticsearch'].get('lex_index', RETRIEVAL_PARAMS['lex']['default_index'])
        )
        return lex_retriever.search(text, embedding, lang, limit)
    except Exception as e:
        print(e)
        return None
