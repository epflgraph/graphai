from graphai.core.common.config import config
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def search_lex(text, embedding=None, lang=None, limit=10, return_embeddings=False):
    try:
        lex_retriever = RETRIEVAL_PARAMS['lex']['retrieval_class'](
            config['elasticsearch'],
            index=config['elasticsearch'].get('lex_index', RETRIEVAL_PARAMS['lex']['default_index'])
        )
        results = lex_retriever.search(text, embedding,
                                       lang=lang,
                                       limit=limit, return_embeddings=return_embeddings)
        return {
            'n_results': len(results),
            'result': results,
            'successful': True
        }
    except Exception as e:
        print(e)
        return {
            'n_results': 0,
            'result': [{'error': str(e)}],
            'successful': False
        }


def search_servicedesk(text, embedding=None, lang=None, cat=None, limit=10, return_embeddings=False):
    try:
        servicedesk_retriever = RETRIEVAL_PARAMS['servicedesk']['retrieval_class'](
            config['elasticsearch'],
            index=config['elasticsearch'].get('servicedesk_index', RETRIEVAL_PARAMS['servicedesk']['default_index'])
        )
        results = servicedesk_retriever.search(text, embedding,
                                               lang=lang, category=cat,
                                               limit=limit, return_embeddings=return_embeddings)
        return {
            'n_results': len(results),
            'result': results,
            'successful': True
        }
    except Exception as e:
        print(e)
        return {
            'n_results': 0,
            'result': [{'error': str(e)}],
            'successful': False
        }



def chunk_text(text, chunk_size=400, chunk_overlap=100):
    # text can be a string or an int to str dict
    if isinstance(text, str):
        full_content = text
    else:
        keys = sorted(list(text.keys()))
        full_content = ' '.join(text[k] for k in keys)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4",
                                                                         chunk_size=chunk_size,
                                                                         chunk_overlap=chunk_overlap)
    split_content = text_splitter.split_text(full_content)
    return {
        'split': split_content,
        'full': full_content
    }
