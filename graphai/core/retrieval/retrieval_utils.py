from graphai.core.common.config import config
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def search_es_index(retriever_type, text, embedding=None, limit=10, return_embeddings=False, **kwargs):
    try:
        retriever = RETRIEVAL_PARAMS[retriever_type]['retrieval_class'](
            config['elasticsearch'],
            index=config['elasticsearch'].get(
                f'{retriever_type}_index', RETRIEVAL_PARAMS[retriever_type]['default_index']
            )
        )
        kwargs = {k: v for k, v in kwargs.items() if k in RETRIEVAL_PARAMS[retriever_type]['filters']}
        results = retriever.search(text, embedding,
                                   limit=limit, return_embeddings=return_embeddings,
                                   return_scores=False,
                                   **kwargs)
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


def retrieve_from_es(embedding_results, text, index_to_search_in, filters=None, limit=10):
    if filters is None:
        filters = dict()
    if index_to_search_in in RETRIEVAL_PARAMS.keys():
        return search_es_index(index_to_search_in,
                               text,
                               embedding_results['result'] if embedding_results['successful'] else None,
                               limit,
                               **filters)
    else:
        return {
            'n_results': 0,
            'result': [{'error': f'Index "{index_to_search_in}" does not exist.'}],
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
