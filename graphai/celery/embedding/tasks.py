from celery import shared_task

from graphai.core.common.fingerprinting import (
    compute_text_fingerprint
)
from graphai.core.common.caching import (
    EmbeddingDBCachingManager
)
from graphai.core.common.lookup import fingerprint_cache_lookup
from graphai.core.embedding.embedding import (
    copy_embedding_object,
    embedding_text_list_embed_parallel,
    EMBEDDING_UNLOAD_WAITING_PERIOD,
    EmbeddingModels,
    compute_embedding_text_fingerprint_callback,
    token_based_embedding_lookup,
    fingerprint_based_embedding_lookup,
    embed_text,
    insert_embedding_into_db,
    jsonify_embedding_results,
    embedding_text_list_fingerprint_parallel,
    embedding_text_list_dummy_fingerprint_parallel,
    embedding_text_list_fingerprint_callback,
    embedding_text_list_embed_jsonify_callback,
    embedding_text_list_embed_callback
)
from graphai.core.common.config import config
from graphai.core.common.common_utils import strtobool

embedding_models = EmbeddingModels()


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.init_embedding', ignore_result=False,
             embedding_obj=embedding_models)
def embedding_init_task(self):
    print('Start init_embedding task')

    if strtobool(config['preload'].get('embedding', 'no')):
        print('Loading embedding models...')
        self.embedding_obj.load_models(load_heavies=False)
    else:
        print('Skipping preloading for embedding models.')

    print('Initializing db caching managers...')
    EmbeddingDBCachingManager(initialize_database=True)

    return True


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_embedding_text', ignore_result=False)
def cache_lookup_embedding_text_fingerprint_task(self, token):
    return fingerprint_cache_lookup(token, EmbeddingDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_embedding_text', ignore_result=False)
def compute_embedding_text_fingerprint_task(self, token, text):
    return compute_text_fingerprint(token, text)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_embedding_text_callback', ignore_result=False)
def compute_embedding_text_fingerprint_callback_task(self, results, text, model_type):
    return compute_embedding_text_fingerprint_callback(results, text, model_type)


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
    if model_type is None:
        return {
            'result': None,
            'successful': False,
            'text_too_large': False,
            'fresh': False,
            'model_type': model_type,
            'device': None
        }
    current_embedding_obj, _ = copy_embedding_object(self.embedding_obj, model_type)
    return embed_text(current_embedding_obj, text, model_type)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embed_text_callback', ignore_result=False)
def embed_text_callback_task(self, results, token, text, model_type, force=False):
    return insert_embedding_into_db(results, token, text, model_type, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embed_text_jsonify_callback', ignore_result=False)
def embed_text_jsonify_callback_task(self, results):
    return jsonify_embedding_results(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embedding_text_list_fingerprint_parallel', ignore_result=False)
def embedding_text_list_fingerprint_parallel_task(self, tokens, text_list, i, n):
    return embedding_text_list_fingerprint_parallel(tokens, text_list, i, n)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embedding_text_list_dummy_fingerprint_parallel', ignore_result=False)
def embedding_text_list_dummy_fingerprint_parallel_task(self, tokens, text_list, i, n):
    return embedding_text_list_dummy_fingerprint_parallel(tokens, text_list, i, n)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embedding_text_list_fingerprint_callback', ignore_result=False)
def embedding_text_list_fingerprint_callback_task(self, results, model_type):
    return embedding_text_list_fingerprint_callback(results, model_type)


@shared_task(bind=True, autoretry_for=(RuntimeError,), retry_backoff=True,
             retry_kwargs={"max_retries": 3, "countdown": 3.0},
             name='text_6.embedding_text_list_embed_parallel', embedding_obj=embedding_models, ignore_result=False)
def embedding_text_list_embed_parallel_task(self, input_list, model_type, i, n, force=False):
    return embedding_text_list_embed_parallel(input_list, self.embedding_obj, model_type, i, n, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embedding_text_list_jsonify_callback', ignore_result=False)
def embedding_text_list_jsonify_callback_task(self, results):
    return embedding_text_list_embed_jsonify_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.embedding_text_list_db_callback', ignore_result=False)
def embedding_text_list_embed_callback_task(self, results, model_type, force=False):
    return embedding_text_list_embed_callback(results, model_type, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.clean_up_large_embedding_objects', embedding_obj=embedding_models, ignore_result=False)
def cleanup_large_embedding_objects_task(self):
    return self.embedding_obj.unload_model(EMBEDDING_UNLOAD_WAITING_PERIOD)
