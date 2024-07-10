from celery import shared_task
import time
import gc

from graphai.api.common.embedding import embedding_models

from graphai.core.common.fingerprinting import perceptual_hash_text
from graphai.core.common.common_utils import get_current_datetime
from graphai.core.interfaces.caching import EmbeddingDBCachingManager
from graphai.core.embedding.embedding import (
    embedding_to_json
)


LONG_TEXT_ERROR = "Text over token limit for selected model (%d)."
UNLOAD_WAITING_PERIOD = 60


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


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.embedding_text_lookup_using_fingerprint', ignore_result=False)
def cache_lookup_embedding_text_using_fingerprint_task(self, token, fp, model_type):
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


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_embedding_text', ignore_result=False)
def cache_lookup_embedding_text_task(self, token):
    db_manager = EmbeddingDBCachingManager()
    existing = db_manager.get_details(token, ['embedding', 'model_type'], using_most_similar=False)[0]
    if existing is not None and existing['embedding'] is not None:
        print('Returning cached result')
        return {
            'result': existing['embedding'],
            'successful': True,
            'text_too_large': False,
            'fresh': False,
            'model_type': existing['model_type'],
            'device': None
        }
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embed_text', embedding_obj=embedding_models, ignore_result=False)
def embed_text_task(self, text, model_type):
    try:
        embedding, text_too_large, model_max_tokens = self.embedding_obj.embed(text, model_type)
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
        'device': self.embedding_obj.get_device()
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embed_text_callback', ignore_result=False)
def embed_text_callback_task(self, results, token, text, model_type, force=False):
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
    elif not results['successful']:
        # in case we fingerprinted something and then failed to embed it, we delete its cache row
        db_manager.delete_cache_rows([token])

    # If the computation was successful, we need to JSONify the resulting numpy array.
    if results['successful']:
        results['result'] = embedding_to_json(results['result'])
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.clean_up_large_objects', embedding_obj=embedding_models, ignore_result=False)
def cleanup_large_embedding_objects_task(self):
    last_heavy_model_use = self.embedding_obj.get_last_usage()
    current_time = time.time()
    result = None
    if current_time - last_heavy_model_use > UNLOAD_WAITING_PERIOD:
        result = self.embedding_obj.unload_heavy_models()
        if len(result) > 0:
            gc.collect()
    return result
