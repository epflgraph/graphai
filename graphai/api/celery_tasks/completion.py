import json

from celery import shared_task
from graphai.core.common.text_utils import force_dict_to_text, ChatGPTSummarizer, find_best_slide_subset
from graphai.core.common.common_utils import get_current_datetime
from graphai.core.common.caching import CompletionDBCachingManager
from graphai.core.text.keywords import get_keywords
from graphai.api.celery_tasks.common import compute_text_fingerprint_common, fingerprint_lookup_retrieve_from_db, \
    fingerprint_lookup_parallel, fingerprint_lookup_direct, fingerprint_lookup_callback


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_completion_text', ignore_result=False)
def compute_summarization_text_fingerprint_task(self, token, text, force=False):
    db_manager = CompletionDBCachingManager()
    return compute_text_fingerprint_common(db_manager, token, force_dict_to_text(text), force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_completion_text_callback', ignore_result=False)
def compute_summarization_text_fingerprint_callback_task(self, results, text, text_type,
                                                         summary_type):
    if results['fresh']:
        token = results['fp_token']
        db_manager = CompletionDBCachingManager()
        values_dict = {
            'fingerprint': results['result'],
            'input_text': force_dict_to_text(text),
            'input_type': text_type,
            'completion_type': summary_type
        }
        existing = db_manager.get_details(token, ['date_added'], using_most_similar=False)[0]
        if existing is None or existing['date_added'] is None:
            current_datetime = get_current_datetime()
            values_dict['date_added'] = current_datetime
        db_manager.insert_or_update_details(token, values_dict)
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.completion_text_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def summarization_text_fingerprint_find_closest_retrieve_from_db_task(self, results, equality_conditions):
    db_manager = CompletionDBCachingManager()
    return fingerprint_lookup_retrieve_from_db(results, db_manager, equality_conditions=equality_conditions)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.completion_text_fingerprint_find_closest_parallel', ignore_result=False)
def summarization_text_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, equality_conditions,
                                                              min_similarity=1):
    db_manager = CompletionDBCachingManager()
    # The equality conditions make sure that the fingerprint lookup happens only among the cached texts with
    # the same summary_type. This is because we don't want a request for the title of a given text to return a
    # hit on the summary of the same text (and vice versa).
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, db_manager, data_type='text',
                                       equality_conditions=equality_conditions)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.completion_text_fingerprint_find_closest_direct', ignore_result=False)
def summarization_text_fingerprint_find_closest_direct_task(self, results, equality_conditions):
    db_manager = CompletionDBCachingManager()
    return fingerprint_lookup_direct(results, db_manager, equality_conditions=equality_conditions)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.completion_text_fingerprint_find_closest_callback', ignore_result=False)
def summarization_text_fingerprint_find_closest_callback_task(self, results_list):
    db_manager = CompletionDBCachingManager()
    return fingerprint_lookup_callback(results_list, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.retrieve_completion_text_fingerprint_final_callback', ignore_result=False)
def summarization_retrieve_text_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    results_to_return = results['fp_results']
    results_to_return['closest'] = results['closest']
    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.completion_text_db_lookup', ignore_result=False)
def lookup_text_completion_task(self, token, text, force=False):
    s = None
    if not force:
        db_manager = CompletionDBCachingManager()
        # The token is [text md5]_[text type]_[summary type]
        all_existing = db_manager.get_details(token, cols=['completion', 'is_json'], using_most_similar=True)
        for existing in all_existing:
            if existing is not None:
                if existing['completion'] is not None:
                    if existing['is_json'] == 1:
                        s = json.loads(existing['completion'])
                    else:
                        s = existing['completion']
                    break
    return {
        'token': token,
        'text': text,
        'original_text': force_dict_to_text(text),
        'existing_results': s
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarize_text_get_keywords', ignore_result=False)
def get_keywords_for_summarization_task(self, input_dict):
    existing_results = input_dict['existing_results']
    text = input_dict['text']
    if existing_results is not None or text is None or len(text) == 0:
        return input_dict
    if isinstance(text, dict):
        new_text = {k: ', '.join(get_keywords(v)) for k, v in text.items()}
    else:
        new_text = ', '.join(get_keywords(text))
    if len(new_text) > 0:
        input_dict['text'] = new_text
    return input_dict


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.completion_text_db_callback', ignore_result=False)
def completion_text_callback_task(self, results, force=False):
    db_manager = CompletionDBCachingManager()
    token = results['token']
    original_text = results['original_text']
    completion = results['result']
    completion_type = results['result_type']
    text_type = results['text_type']
    n_tokens_total = results['n_tokens_total']
    if results['fresh']:
        if isinstance(completion, dict):
            completion = json.dumps(completion)
            is_json = 1
        else:
            is_json = 0
        values_dict = {
            'input_text': original_text,
            'completion': completion,
            'completion_type': completion_type,
            'input_type': text_type,
            'completion_length': len(completion.split(' ')),
            'completion_token_total': n_tokens_total['total_tokens'],
            'completion_cost': n_tokens_total['cost'],
            'is_json': is_json
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
            # Only happens if the other token has been fingerprinted first without being translated.
            closest = db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                db_manager.insert_or_update_details(
                    closest, values_dict
                )
    elif not results['successful']:
        # If the summarization wasn't successful, we delete the cache row because it serves no other purpose
        db_manager.delete_cache_rows([token])
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.completion_text_chatgpt_compute', ignore_result=False)
def request_text_completion_task(self, token_and_text, text_type='text', result_type='cleanup', debug=False):
    assert result_type in ['summary', 'cleanup']
    existing_results = token_and_text['existing_results']
    token = token_and_text['token']
    text = token_and_text['text']
    original_text = token_and_text.get('original_text', text)
    if text is None or len(text) == 0:
        result_dict = {
            'token': token,
            'text': text,
            'original_text': original_text,
            'result': None,
            'result_type': None,
            'text_type': None,
            'fresh': False,
            'successful': False,
            'too_many_tokens': False,
            'n_tokens_total': None,
            'full_message': None
        }
        return result_dict
    if existing_results is not None:
        return {
            'token': token,
            'text': text,
            'original_text': original_text,
            'result': existing_results,
            'result_type': result_type,
            'text_type': text_type,
            'fresh': False,
            'successful': True,
            'too_many_tokens': False,
            'n_tokens_total': None,
            'full_message': None
        }
    summarizer = ChatGPTSummarizer()
    if result_type == 'cleanup':
        # Cleanup
        results, message, too_many_tokens, n_tokens_total = summarizer.cleanup_text(
            text, text_type=text_type, handwriting=True)
        if results is not None:
            results = {'subject': results['subject'], 'text': results['cleaned'],
                       'for_wikify': f'{results["subject"]}\n\n{results["cleaned"]}'}
    else:
        fields = ['summary_long', 'summary_short', 'title']
        # Summary
        if text_type == 'lecture':
            fn = summarizer.summarize_lecture
        elif text_type == 'academic_entity':
            fn = summarizer.summarize_academic_entity
            fields += ['top_3_categories', 'inferred_subtype']
        else:
            fn = summarizer.summarize_generic
        results, message, too_many_tokens, n_tokens_total = fn(text)
        if results is not None:
            results = {field: results[field] for field in fields}
    if not debug:
        message = None
    return {
        'token': token,
        'text': text,
        'original_text': original_text,
        'result': results,
        'result_type': result_type,
        'text_type': text_type,
        'fresh': results is not None,
        'successful': results is not None,
        'too_many_tokens': too_many_tokens,
        'n_tokens_total': n_tokens_total,
        'full_message': message
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.cleanup_text_chatgpt_simulate', ignore_result=False)
def simulate_completion_task(self, text, text_type='text', result_type='cleanup'):
    summarizer = ChatGPTSummarizer()
    if result_type == 'cleanup':
        # cleanup
        _, system_message, too_many_tokens, token_count = summarizer.cleanup_text(
            text, text_type=text_type, handwriting=True, simulate=True)
    else:
        # summary
        if text_type == 'lecture':
            _, system_message, too_many_tokens, token_count = summarizer.summarize_lecture(
                text, simulate=True)
        elif text_type == 'academic_entity':
            _, system_message, too_many_tokens, token_count = summarizer.summarize_academic_entity(
                text, simulate=True)
        else:
            _, system_message, too_many_tokens, token_count = summarizer.summarize_generic(
                text, simulate=True)
    return {
        'result': None,
        'result_type': result_type,
        'fresh': False,
        'successful': True,
        'too_many_tokens': too_many_tokens,
        'n_tokens_total': token_count,
        'full_message': system_message
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.choose_best_subset', ignore_result=False)
def choose_best_subset_task(self, slide_number_to_concepts, coverage=1.0, min_freq=2):
    slide_numbers = sorted(list(slide_number_to_concepts.keys()))
    slide_concept_list = [slide_number_to_concepts[n] for n in slide_numbers]
    cover, best_indices = find_best_slide_subset(slide_concept_list, coverage, True, min_freq)
    return {
        'subset': [slide_numbers[i] for i in best_indices]
    }
