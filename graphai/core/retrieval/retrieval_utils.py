from datetime import datetime
from graphai.core.common.config import config
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS
from langchain_text_splitters import RecursiveCharacterTextSplitter


INSUFFICIENT_ACCESS_ERROR = {
    'n_results': 0,
    'result': [{'error': 'You do not have access to the selected index.'}],
    'successful': False
}


def normalize_index_name(index_name):
    return index_name.lower().replace(' ', '_').replace('-', '')


def has_invalid_characters(index_name):
    invalid_characters = '\\/*?"<>| ,;:#'
    for char in invalid_characters:
        if char in index_name:
            return True
    return False


def search_es_index(retriever_type,
                    index_name,
                    allowed_filters,
                    text,
                    embedding=None,
                    limit=10,
                    return_embeddings=False,
                    return_scores=False,
                    **kwargs):
    try:
        retriever = retriever_type(
            config['elasticsearch'],
            index=index_name
        )
        if allowed_filters is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed_filters}
        results = retriever.search(text, embedding,
                                   limit=limit, return_embeddings=return_embeddings,
                                   return_scores=return_scores,
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


def retrieve_from_es(embedding_results, text, index_to_search_in,
                     filters=None, limit=10, return_scores=False, filter_by_date=False):
    if filters is None:
        filters = dict()
    if filter_by_date:
        right_now = datetime.now().isoformat()
        filters = filters | {
            'from_date': {'lte': right_now},
            'until_date': {'gte': right_now}
        }
    if index_to_search_in in RETRIEVAL_PARAMS.keys():
        return search_es_index(
            retriever_type=RETRIEVAL_PARAMS[index_to_search_in]['retrieval_class'],
            index_name=config['elasticsearch'].get(
                f'{index_to_search_in}_index', RETRIEVAL_PARAMS[index_to_search_in]['default_index']
            ),
            allowed_filters=RETRIEVAL_PARAMS[index_to_search_in]['filters'],
            text=text,
            embedding=embedding_results['result'] if embedding_results['successful'] else None,
            limit=limit,
            return_scores=return_scores,
            **filters
        )
    else:
        index_to_search_in = normalize_index_name(index_to_search_in)
        if has_invalid_characters(index_to_search_in):
            return {
                'n_results': 0,
                'result': [{'error': f'Index name {index_to_search_in} contains invalid characters.'}],
                'successful': False
            }
        return search_es_index(
            retriever_type=RETRIEVAL_PARAMS['default']['retrieval_class'],
            index_name=config['elasticsearch'].get(
                f'{index_to_search_in}_index', RETRIEVAL_PARAMS['default']['default_index'] % index_to_search_in
            ),
            allowed_filters=None,
            text=text,
            embedding=embedding_results['result'] if embedding_results['successful'] else None,
            limit=limit,
            return_scores=return_scores,
            **filters
        )


def chunk_text(text, chunk_size=400, chunk_overlap=100,
               one_chunk_per_page=False, one_chunk_per_doc=False):
    # text can be a string or an int to str dict
    if isinstance(text, str):
        keys = None
        full_content = text
    else:
        text = {int(k): text[k] for k in text}
        keys = sorted(list(text.keys()))
        full_content = ' '.join(text[k] for k in keys)
    if one_chunk_per_doc:
        split_content = [full_content]
    elif one_chunk_per_page and keys is not None:
        # If 1 chunk/page is enabled but the original text is one block with no pages, we can't do 1 chunk/page
        split_content = [text[k] for k in keys]
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4",
                                                                             chunk_size=chunk_size,
                                                                             chunk_overlap=chunk_overlap)
        split_content = text_splitter.split_text(full_content)
    return {
        'split': split_content,
        'full': full_content
    }
