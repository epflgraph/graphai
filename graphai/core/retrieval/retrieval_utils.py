from graphai.core.common.config import config
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS


def search_lex(text, embedding=None, lang=None, limit=10, return_embeddings=False):
    try:
        lex_retriever = RETRIEVAL_PARAMS['lex']['retrieval_class'](
            config['elasticsearch'],
            index=config['elasticsearch'].get('lex_index', RETRIEVAL_PARAMS['lex']['default_index'])
        )
        results = lex_retriever.search(text, embedding, lang, limit=limit, return_embeddings=return_embeddings)
        return {
            'n_results': len(results),
            'result': results,
            'successful': True
        }
    except Exception as e:
        print(e)
        return {
            'n_results': 0,
            'result': None,
            'successful': False
        }
