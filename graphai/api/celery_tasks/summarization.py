from celery import shared_task
from graphai.core.common.video import ChatGPTSummarizer, get_current_datetime
from graphai.core.common.caching import SummaryDBCachingManager
from graphai.core.text.keywords import get_keywords
from graphai.api.celery_tasks.common import compute_text_fingerprint_common, fingerprint_lookup_retrieve_from_db, \
    fingerprint_lookup_parallel, fingerprint_lookup_direct, fingerprint_lookup_callback


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_summarization_text', ignore_result=False)
def compute_summarization_text_fingerprint_task(self, token, text, force=False):
    db_manager = SummaryDBCachingManager()
    return compute_text_fingerprint_common(db_manager, token, text, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_summarization_text_callback', ignore_result=False)
def compute_summarization_text_fingerprint_callback_task(self, results, text, summary_type):
    if results['fresh']:
        token = results['fp_token']
        db_manager = SummaryDBCachingManager()
        values_dict = {
            'fingerprint': results['result'],
            'input_text': text,
            'summary_type': summary_type
        }
        existing = db_manager.get_details(token, ['date_added'], using_most_similar=False)[0]
        if existing is None or existing['date_added'] is None:
            current_datetime = get_current_datetime()
            values_dict['date_added'] = current_datetime
        db_manager.insert_or_update_details(token, values_dict)
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarization_text_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def summarization_text_fingerprint_find_closest_retrieve_from_db_task(self, results, equality_conditions):
    db_manager = SummaryDBCachingManager()
    return fingerprint_lookup_retrieve_from_db(results, db_manager, equality_conditions=equality_conditions)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarization_text_fingerprint_find_closest_parallel', ignore_result=False)
def summarization_text_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, equality_conditions,
                                                              min_similarity=1):
    db_manager = SummaryDBCachingManager()
    # The equality conditions make sure that the fingerprint lookup happens only among the cached texts with
    # the same summary_type. This is because we don't want a request for the title of a given text to return a
    # hit on the summary of the same text (and vice versa).
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, db_manager, data_type='text',
                                       equality_conditions=equality_conditions)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarization_text_fingerprint_find_closest_direct', ignore_result=False)
def summarization_text_fingerprint_find_closest_direct_task(self, results, equality_conditions):
    db_manager = SummaryDBCachingManager()
    return fingerprint_lookup_direct(results, db_manager, equality_conditions=equality_conditions)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarization_text_fingerprint_find_closest_callback', ignore_result=False)
def summarization_text_fingerprint_find_closest_callback_task(self, results_list):
    db_manager = SummaryDBCachingManager()
    return fingerprint_lookup_callback(results_list, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.retrieve_summarization_text_fingerprint_final_callback', ignore_result=False)
def summarization_retrieve_text_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    results_to_return = results['fp_results']
    results_to_return['closest'] = results['closest']
    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarize_text_db_lookup', ignore_result=False)
def lookup_text_summary_task(self, token, text, force=False):
    if not force:
        db_manager = SummaryDBCachingManager()
        # The token is [text md5]_[summary type]
        # We retrieve both the summary and the summary_type because both are needed for the results at the end
        all_existing = db_manager.get_details(token, cols=['summary', 'summary_type'], using_most_similar=True)
        for existing in all_existing:
            if existing is not None:
                if existing['summary'] is not None:
                    return {
                        'token': token,
                        'text': text,
                        'existing_results': {
                            'summary': existing['summary'],
                            'summary_type': existing['summary_type']
                        }
                    }
    return {
        'token': token,
        'text': text,
        'existing_results': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarize_text_get_keywords', ignore_result=False)
def get_keywords_for_summarization_task(self, input_dict, use_keywords=True):
    existing_results = input_dict['existing_results']
    text = input_dict['text']
    input_dict['original_text'] = text
    if existing_results is not None or not use_keywords or text is None or len(text) == 0:
        input_dict['is_keywords'] = False
        return input_dict
    keywords = get_keywords(input_dict['text'])
    new_text = ', '.join(keywords)
    if len(new_text) > 0:
        input_dict['text'] = new_text
        input_dict['is_keywords'] = True
    else:
        input_dict['is_keywords'] = False
    return input_dict


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarize_text_chatgpt_compute', ignore_result=False)
def summarize_text_task(self, token_and_text, text_type='lecture', title=False, title_len=20, summary_len=100):
    existing_results = token_and_text['existing_results']
    token = token_and_text['token']
    text = token_and_text['text']
    original_text = token_and_text['original_text']
    if text is None or len(text) == 0:
        return {
            'token': token,
            'text': text,
            'original_text': original_text,
            'summary': None,
            'summary_type': None,
            'fresh': False,
            'successful': False,
            'too_many_tokens': False
        }
    if existing_results is not None:
        return {
            'token': token,
            'text': text,
            'original_text': original_text,
            'summary': existing_results['summary'],
            'summary_type': existing_results['summary_type'],
            'fresh': False,
            'successful': True,
            'too_many_tokens': False
        }
    is_keywords = token_and_text.get('keyword', False)
    summarizer = ChatGPTSummarizer()
    results, too_many_tokens = summarizer.generate_summary(
        text, text_type=text_type, keywords=is_keywords, ordered=not is_keywords,
        title=title, max_len=title_len if title else summary_len)
    return {
        'token': token,
        'text': text,
        'original_text': original_text,
        'summary': results,
        'summary_type': 'title' if title else 'summary',
        'fresh': results is not None,
        'successful': results is not None,
        'too_many_tokens': too_many_tokens
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarize_text_db_callback', ignore_result=False)
def summarize_text_callback_task(self, results, force=False):
    db_manager = SummaryDBCachingManager()
    token = results['token']
    original_text = results['original_text']
    summary = results['summary']
    summary_type = results['summary_type']
    if results['fresh']:
        values_dict = {
            'input_text': original_text,
            'summary': summary,
            'summary_type': summary_type,
            'summary_length': len(summary.split(' '))
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
