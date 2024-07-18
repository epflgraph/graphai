import copy

from celery import shared_task
from itertools import chain

from graphai.api.common.embedding import embedding_models

from graphai.core.common.fingerprinting import perceptual_hash_text
from graphai.core.common.common_utils import get_current_datetime
from graphai.core.interfaces.caching import EmbeddingDBCachingManager
from graphai.core.embedding.embedding import (
    embedding_to_json,
    get_text_token_count_using_model,
    EMBEDDING_UNLOAD_WAITING_PERIOD
)


LONG_TEXT_ERROR = "Text over token limit for selected model (%d)."


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_embedding_text', ignore_result=False)
def cache_lookup_embedding_text_fingerprint_task(self, token):
    db_manager = EmbeddingDBCachingManager()
    existing = db_manager.get_details(token, cols=['fingerprint'])[0]
    if existing is not None and existing['fingerprint'] is not None:
        existing_closest = db_manager.get_closest_match(token)
        return {
            'result': existing['fingerprint'],
            'closest': existing_closest,
            'fresh': False
        }
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_embedding_text', ignore_result=False)
def compute_embedding_text_fingerprint_task(self, token, text):
    fp = perceptual_hash_text(text)

    return {
        'result': fp,
        'token': token,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_embedding_text_callback', ignore_result=False)
def compute_embedding_text_fingerprint_callback_task(self, results, text, model_type):
    # This task does not have the condition of the 'fresh' flag being True because text fingerprinting never fails
    fp = results['result']
    token = results['token']
    db_manager = EmbeddingDBCachingManager()
    values_dict = {
        'fingerprint': fp,
        'source': text,
        'model_type': model_type,
        'date_added': get_current_datetime()
    }
    db_manager.insert_or_update_details(token, values_dict)

    return results


def token_based_embedding_lookup(token, model_type):
    db_manager = EmbeddingDBCachingManager()
    existing = db_manager.get_details(token, ['embedding'], using_most_similar=False)[0]
    if existing is not None and existing['embedding'] is not None:
        print('Returning cached result')
        return {
            'result': existing['embedding'],
            'successful': True,
            'text_too_large': False,
            'fresh': False,
            'model_type': model_type,
            'device': None
        }
    return None


def fingerprint_based_embedding_lookup(token, fp, model_type):
    db_manager = EmbeddingDBCachingManager()
    # Super quick fingerprint lookup
    closest_embedding = db_manager.get_all_details(['embedding', 'model_type'], allow_nulls=False,
                                                   equality_conditions={'fingerprint': fp,
                                                                        'model_type': model_type
                                                                        })
    if closest_embedding is not None:
        all_keys = list(closest_embedding.keys())
        embedding_json = closest_embedding[all_keys[0]]['embedding']
        db_manager.insert_or_update_details(token, {
            'embedding': embedding_json
        })
        return {
            'result': embedding_json,
            'successful': True,
            'text_too_large': False,
            'fresh': False,
            'model_type': model_type,
            'device': None
        }
    return None


def embed_text(models, text, model_type):
    try:
        embedding, text_too_large, model_max_tokens = models.embed(text, model_type)
        success = embedding is not None
        if text_too_large:
            embedding = LONG_TEXT_ERROR % model_max_tokens
    except NotImplementedError as e:
        print(e)
        embedding = str(e)
        success = False
        text_too_large = False

    return {
        'result': embedding,
        'successful': success,
        'text_too_large': text_too_large,
        'fresh': success,
        'model_type': model_type,
        'device': models.get_device()
    }


def insert_embedding_into_db(results, token, text, model_type, force=False):
    db_manager = EmbeddingDBCachingManager()
    if results['fresh']:
        values_dict = {
            'source': text,
            'embedding': embedding_to_json(results['result']),
            'model_type': model_type
        }
        existing = db_manager.get_details(token, ['date_added'], using_most_similar=False)[0]
        if existing is None or existing['date_added'] is None:
            current_datetime = get_current_datetime()
            values_dict['date_added'] = current_datetime
        # Inserting values for original token
        db_manager.insert_or_update_details(
            token, values_dict
        )
        if not force:
            # Inserting the same values for closest token if different than original token
            # Only happens if the other token has been fingerprinted first without having its embedding calculated.
            closest = db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                db_manager.insert_or_update_details(
                    closest, values_dict
                )
        # If the computation is fresh, we need to JSONify the resulting numpy array.
        # Non-fresh successful computation results come from cache hits, and those are already in JSON.
        # Non-successful computation results are normal strings.
        results['result'] = embedding_to_json(results['result'])
    elif not results['successful']:
        # in case we fingerprinted something and then failed to embed it, we delete its cache row
        db_manager.delete_cache_rows([token])
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.embedding_text_lookup_using_fingerprint', ignore_result=False)
def cache_lookup_embedding_text_using_fingerprint_task(self, token, fp, model_type):
    return fingerprint_based_embedding_lookup(token, fp, model_type)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_embedding_text', ignore_result=False)
def cache_lookup_embedding_text_task(self, token, model_type):
    return token_based_embedding_lookup(token, model_type)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embed_text', embedding_obj=embedding_models, ignore_result=False)
def embed_text_task(self, text, model_type):
    return embed_text(self.embedding_obj, text, model_type)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embed_text_callback', ignore_result=False)
def embed_text_callback_task(self, results, token, text, model_type, force=False):
    return insert_embedding_into_db(results, token, text, model_type, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embedding_text_list_fingerprint_parallel', ignore_result=False)
def embedding_text_list_fingerprint_parallel_task(self, tokens, text_list, i, n):
    start_index = int(i * len(tokens) / n)
    end_index = int((i + 1) * len(tokens) / n)
    if start_index == end_index:
        return []
    tokens = tokens[start_index:end_index]
    text_list = text_list[start_index:end_index]
    results = list()
    db_manager = EmbeddingDBCachingManager()
    for j in range(len(tokens)):
        existing = db_manager.get_details(tokens[j], cols=['fingerprint'])[0]
        if existing is not None and existing['fingerprint'] is not None:
            results.append(
                {
                    'token': tokens[j],
                    'fp': existing['fingerprint'],
                    'text': text_list[j],
                    'fresh': False
                }
            )
        else:
            results.append(
                {
                    'token': tokens[j],
                    'fp': perceptual_hash_text(text_list[j]),
                    'text': text_list[j],
                    'fresh': True
                }
            )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embedding_text_list_fingerprint_callback', ignore_result=False)
def embedding_text_list_fingerprint_callback_task(self, results, model_type):
    db_manager = EmbeddingDBCachingManager()
    all_results = list(chain.from_iterable(results))
    for result in all_results:
        if result['fresh']:
            values_dict = {
                'fingerprint': result['fp'],
                'source': result['text'],
                'model_type': model_type,
                'date_added': get_current_datetime()
            }
            db_manager.insert_or_update_details(result['token'], values_dict)

    return all_results


@shared_task(bind=True, autoretry_for=(RuntimeError,), retry_backoff=True,
             retry_kwargs={"max_retries": 3, "countdown": 2.0},
             name='text_6.embedding_text_list_embed_parallel', embedding_obj=embedding_models, ignore_result=False)
def embedding_text_list_embed_parallel_task(self, input_list, model_type, i, n, force=False):
    start_index = int(i * len(input_list) / n)
    end_index = int((i + 1) * len(input_list) / n)
    if start_index == end_index:
        return []
    input_list = input_list[start_index:end_index]
    input_list = dict(enumerate(input_list))
    results_dict = dict()
    if not force:
        for ind in input_list:
            current_dict = input_list[ind]
            current_results = token_based_embedding_lookup(current_dict['token'], model_type)
            if current_results is None:
                current_results = fingerprint_based_embedding_lookup(
                    current_dict['token'], current_dict['fp'], model_type
                )
            if current_results is not None:
                current_results['id_token'] = current_dict['token']
                current_results['source'] = current_dict['text']
                results_dict[ind] = current_results
    remaining_indices = list(set(input_list.keys()).difference(set(results_dict.keys())))
    if len(remaining_indices) > 0:
        tokenizer = self.embedding_obj.get_tokenizer(model_type)
        if tokenizer is None:
            raise NotImplementedError(f"Model type {model_type} cannot be found!")
        tokenizer = copy.deepcopy(tokenizer)
        token_counts = [get_text_token_count_using_model(tokenizer, input_list[ind]['text'])
                        for ind in remaining_indices]
        model_max_tokens = self.embedding_obj.get_max_tokens(model_type)
        i = 0
        while i < len(remaining_indices):
            current_token_count_sum = token_counts[i]
            j = i + 1
            while current_token_count_sum < model_max_tokens and j < len(remaining_indices):
                current_token_count_sum += token_counts[j]
                j += 1
            if current_token_count_sum >= model_max_tokens:
                j -= 1
            if j == i:
                j = i + 1
            current_results_list = embed_text(self.embedding_obj,
                                              [input_list[remaining_indices[k]]['text'] for k in range(i, j)],
                                              model_type)
            for k in range(i, j):
                current_results = {
                    'result': current_results_list['result'][k - i]
                    if not isinstance(current_results_list['result'], str)
                    else current_results_list['result'],
                    'successful': current_results_list['successful'],
                    'text_too_large': current_results_list['text_too_large'],
                    'fresh': current_results_list['fresh'],
                    'model_type': current_results_list['model_type'],
                    'device': current_results_list['device'],
                    'id_token': input_list[remaining_indices[k]]['token'],
                    'source': input_list[remaining_indices[k]]['text']
                }
                results_dict[remaining_indices[k]] = current_results
            i = j
    sorted_indices = sorted(results_dict.keys())
    return [results_dict[i] for i in sorted_indices]


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embedding_text_list_embed_callback', ignore_result=False)
def embedding_text_list_embed_callback_task(self, results, model_type, force=False):
    all_results = list(chain.from_iterable(results))
    new_results = list()
    for result in all_results:
        new_result = insert_embedding_into_db(result, result['id_token'], result['source'], model_type, force)
        del new_result['id_token']
        del new_result['source']
        new_results.append(new_result)
    return new_results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.clean_up_large_embedding_objects', embedding_obj=embedding_models, ignore_result=False)
def cleanup_large_embedding_objects_task(self):
    return self.embedding_obj.unload_model(EMBEDDING_UNLOAD_WAITING_PERIOD)
