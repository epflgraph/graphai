from celery import shared_task

from graphai.core.common.video import find_closest_audio_fingerprint_from_list, \
    find_closest_image_fingerprint_from_list


def format_api_results(id, name, status, result):
    return {
        "task_id": id,
        "task_name": name,
        "task_status": status,
        "task_result": result
    }


def fingerprint_lookup_retrieve_from_db(results, token, db_manager, equality_conditions=None):
    target_fingerprint = results['result']
    fresh = results['fresh']
    # If the fingerprint computation has been unsuccessful or if cached results are being returned,
    # then there it is not necessary (or even possible, in the former case) to compute the closest
    # audio fingerprint, so we just pass the fingerprinting results along.
    if target_fingerprint is None or not fresh:
        return {
            'target_fp': None,
            'cache_count': None,
            'fp_results': results
        }
    # Retrieving all the tokens and their fingerprints. Since at least one audio has been extracted
    # (i.e. this one), this result is never null. In addition, there's at least one non-null fingerprint
    # value (again, for the present file).
    n_cache_rows = db_manager.get_cache_count(['fingerprint'], equality_conditions=equality_conditions)

    return {
        'target_fp': target_fingerprint,
        'cache_count': n_cache_rows,
        'fp_results': results
    }


def fingerprint_lookup_parallel(input_dict, token, i, n_total, min_similarity, db_manager, data_type='audio',
                                equality_conditions=None):
    assert data_type in ['audio', 'image', 'text']
    # This parallel task's "closest fingerprint" result is null if either
    # a) the computation has been disabled (indicated by the token list being null), or
    # b) there are no previous fingerprints (indicated by the list of all tokens being empty)
    if input_dict['cache_count'] is None or input_dict['cache_count'] == 0:
        return {
            'closest': None,
            'closest_fp': None,
            'max_score': None,
            'fp_results': input_dict['fp_results']
        }
    # Get the total number of tokens and fingerprints
    n_tokens_all = input_dict['cache_count']
    # Compute the start and end indices
    start_index = int(i / n_total * n_tokens_all)
    end_index = int((i + 1) / n_total * n_tokens_all)
    limit = end_index - start_index
    if limit <= 0:
        return {
            'closest': None,
            'closest_fp': None,
            'max_score': None,
            'fp_results': input_dict['fp_results']
        }
    tokens_and_fingerprints = db_manager.get_all_details(
        ['fingerprint'], start=start_index, limit=limit, exclude_token=token, using_most_similar=False,
        allow_nulls=False, equality_conditions=equality_conditions
    )
    if tokens_and_fingerprints is None or len(tokens_and_fingerprints) == 0:
        return {
            'closest': None,
            'closest_fp': None,
            'max_score': None,
            'fp_results': input_dict['fp_results']
        }
    all_tokens = list(tokens_and_fingerprints.keys())
    all_fingerprints = [tokens_and_fingerprints[key]['fingerprint'] for key in all_tokens]
    # Find the closest token for this batch
    # Note: null fingerprint values are automatically handled and don't need to be filtered out.
    if data_type == 'audio':
        find_closest_func = find_closest_audio_fingerprint_from_list
    else:
        # Text and image fingerprinting are done the same way, so 'image' and 'text' are treated as the same here.
        find_closest_func = find_closest_image_fingerprint_from_list
    closest_token, closest_fingerprint, score = find_closest_func(
        input_dict['target_fp'],
        all_fingerprints,
        all_tokens,
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
        if fp_results['result'] is not None:
            closest = db_manager.get_closest_match(original_token)
        else:
            closest = None
        return {
            'closest': closest,
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
             name='video_2.ignore_fingerprint_results_callback', ignore_result=False)
def ignore_fingerprint_results_callback_task(self, results, results_to_return):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    return results_to_return