from graphai.core.common.video import find_closest_audio_fingerprint_from_list, \
    find_closest_image_fingerprint_from_list
from graphai.api.common.video import transcription_model, local_ocr_nlp_models, google_ocr_model
from graphai.api.common.graph import graph
from graphai.api.common.ontology import ontology
from celery import shared_task

def format_api_results(id, name, status, result):
    return {
        "task_id": id,
        "task_name": name,
        "task_status": status,
        "task_result": result
    }


def fingerprint_lookup_retrieve_from_db(results, token, db_manager):
    target_fingerprint = results['result']
    fresh = results['fresh']
    # If the fingerprint computation has been unsuccessful or if cached results are being returned,
    # then there it is not necessary (or even possible, in the former case) to compute the closest
    # audio fingerprint, so we just pass the fingerprinting results along.
    if target_fingerprint is None or not fresh:
        return {
            'target_fp': None,
            'all_tokens': None,
            'all_fingerprints': None,
            'fp_results': results
        }
    # Retrieving all the tokens and their fingerprints. Since at least one audio has been extracted
    # (i.e. this one), this result is never null. In addition, there's at least one non-null fingerprint
    # value (again, for the present audio file).
    tokens_and_fingerprints = db_manager.get_all_details(['fingerprint'], using_most_similar=False)
    all_tokens = list(tokens_and_fingerprints.keys())
    all_fingerprints = [tokens_and_fingerprints[key]['fingerprint'] for key in all_tokens]
    # Now we remove the token of the current file itself, because otherwise we'd always get the token itself
    # as the most similar token.
    index_to_remove = all_tokens.index(token)
    del all_tokens[index_to_remove]
    del all_fingerprints[index_to_remove]
    return {
        'target_fp': target_fingerprint,
        'all_tokens': all_tokens,
        'all_fingerprints': all_fingerprints,
        'fp_results': results
    }


def fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, data_type='audio'):
    assert data_type in ['audio', 'image']
    # This parallel task's "closest fingerprint" result is null if either
    # a) the computation has been disabled (indicated by the token list being null), or
    # b) there are no previous fingerprints (indicated by the list of all tokens being empty)
    if input_dict['all_tokens'] is None or len(input_dict['all_tokens']) == 0:
        return {
            'closest': None,
            'closest_fp': None,
            'max_score': None,
            'fp_results': input_dict['fp_results']
        }
    # Get the total number of tokens and fingerprints
    n_tokens_all = len(input_dict['all_tokens'])
    # Compute the start and end indices
    start_index = int(i / n_total * n_tokens_all)
    end_index = int((i + 1) / n_total * n_tokens_all)
    # Find the closest token for this batch
    # Note: null fingerprint values are automatically handled and don't need to be filtered out.
    if data_type == 'audio':
        find_closest_func = find_closest_audio_fingerprint_from_list
    else:
        find_closest_func = find_closest_image_fingerprint_from_list
    closest_token, closest_fingerprint, score = find_closest_func(
        input_dict['target_fp'],
        input_dict['all_fingerprints'][start_index:end_index],
        input_dict['all_tokens'][start_index:end_index],
        min_similarity=min_similarity
    )
    return {
        'closest': closest_token,
        'closest_fp': closest_fingerprint,
        'max_score': score,
        'fp_results': input_dict['fp_results']
    }


def fingerprint_lookup_callback(results_list, original_token, db_manager):
    # Passing fingerprinting results along if it's been unsuccessful or a cached result has been returned
    # This is essentially the same check as in all the other find_closest tasks.
    fp_results = results_list[0]['fp_results']
    if fp_results['result'] is None or not fp_results['fresh']:
        return{
            'closest': None,
            'score': None,
            'fp_results': fp_results
        }
    results = [(x['closest'], x['closest_fp'], x['max_score']) for x in results_list]
    results = [x for x in results if x[0] is not None]
    # If all results are null and the list is thus empty, then no closest fingerprint has been found,
    # and therefore, the closest token to this one is itself.
    if len(results) == 0:
        closest_token = original_token
        max_score = -1
    else:
        max_score = max([x[2] for x in results])
        closest_token = [x[0] for x in results if x[2] == max_score][0]
        closest_token = db_manager.resolve_most_similar_chain(closest_token)
    # Whether the closest token is itself or another token, we store the result in the database.
    db_manager.insert_or_update_closest_match(
        original_token,
        {
            'most_similar_token': closest_token
        }
    )
    return {'closest': closest_token, 'score': max_score, 'fp_results': fp_results}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.dummy_task', ignore_result=False)
def dummy_task(self, results):
    # This task is required for chaining groups due to the peculiarities of celery
    # Whenever there are two groups in one chain of tasks, there need to be at least
    # TWO tasks between them, and this dummy task is simply an f(x)=x function.
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='common.lazy_load_all', ignore_result=False,
             graph_obj=graph, ontology_obj=ontology, transcription_obj=transcription_model,
             nlp_obj=local_ocr_nlp_models, google_obj=google_ocr_model)
def lazy_loader_task(self):
    # This task force-loads all the lazy-loading objects in the celery process
    print('Starting force-load...')
    self.graph_obj.fetch_from_db()
    print('Graph tables loaded')
    self.ontology_obj.fetch_from_db()
    print('Ontology tables loaded')
    self.transcription_obj.load_model_whisper()
    print('Transcription model loaded')
    self.nlp_obj.get_nlp_models()
    print('NLP models loaded')
    self.google_obj.establish_connection()
    print('Google API connection established')
    return True
