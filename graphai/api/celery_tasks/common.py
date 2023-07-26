from celery import shared_task

from graphai.core.common.video import find_closest_audio_fingerprint_from_list, \
    find_closest_image_fingerprint_from_list, perceptual_hash_text


def format_api_results(id, name, status, result):
    """
    Formats results coming from celery into the common output format of the API
    Args:
        id: Id of the task
        name: Name of the task
        status: Task status
        result: Task results

    Returns:
        Appropriately formatted results dictionary
    """
    return {
        "task_id": id,
        "task_name": name,
        "task_status": status,
        "task_result": result
    }


def compute_text_fingerprint_common(db_manager, token, text, force=False):
    existing = db_manager.get_details(token, cols=['fingerprint'])[0]

    if existing is not None and not force:
        if existing['fingerprint'] is not None:
            return {
                'result': existing['fingerprint'],
                'fp_token': existing['id_token'],
                'perform_lookup': False,
                'fresh': False
            }

    fp = perceptual_hash_text(text)

    return {
        'result': fp,
        'fp_token': token,
        'perform_lookup': True,
        'fresh': True
    }


def fingerprint_lookup_retrieve_from_db(results, db_manager, equality_conditions=None):
    """
    Retrieves the number of cache rows relevant to fingerprint lookup
    Args:
        results: Dict containing results of fingerprint computation
        db_manager: DBCachingManagerBase object
        equality_conditions: Dictionary of equality conditions for fingerprint counting. Only used for translation,
                where the lookup needs to be done only among cached rows with the same source and target langs.

    Returns:
        Dict of original results plus the number of relevant cache rows.
    """
    target_fingerprint = results['result']

    # If the fingerprint computation has been unsuccessful or if cached results are being returned,
    # then there it is not necessary (or even possible, in the former case) to compute the closest
    # audio fingerprint, so we just pass the fingerprinting results along.
    if not results['perform_lookup']:
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


def fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, db_manager, data_type='audio',
                                equality_conditions=None):
    """
    Performs parallel lookup of fingerprints
    Args:
        input_dict: Dict of results from previous step
        i: The index of this parallel task
        n_total: Total number of parallel tasks
        min_similarity: Minimum similarity threshold for two fingerprints to be considered a match
        db_manager: DBCachingManagerBase object
        data_type: Type of data. Can be 'audio', 'video', 'image', and 'text'
        equality_conditions: Equality conditions dict, only used for translation ('text' mode)

    Returns:
        Dict with fingerprinting results, plus details of the closest match (which can be None)
    """
    assert data_type in ['audio', 'image', 'text', 'video']
    # This parallel task's "closest fingerprint" result is null if either
    # a) the computation has been disabled (indicated by the token list being null), or
    # b) there are no previous fingerprints (indicated by the list of all tokens being empty)
    fp_results = input_dict['fp_results']
    if not fp_results['perform_lookup'] or input_dict['cache_count'] is None or input_dict['cache_count'] == 0:
        return {
            'closest': None,
            'closest_fp': None,
            'closest_date': None,
            'max_score': None,
            'fp_results': fp_results
        }
    token = fp_results['fp_token']
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
            'closest_date': None,
            'max_score': None,
            'fp_results': fp_results
        }
    # The resulting list will be sorted by date_added (asc)
    # Nulls are not allowed because if a row doesn't have a fingerprint, it makes no sense to include it in a
    # fingerprint lookup.
    tokens_and_fingerprints = db_manager.get_all_details(
        ['fingerprint', 'date_added'], start=start_index, limit=limit, exclude_token=token,
        allow_nulls=False, equality_conditions=equality_conditions
    )

    if tokens_and_fingerprints is None or len(tokens_and_fingerprints) == 0:
        return {
            'closest': None,
            'closest_fp': None,
            'closest_date': None,
            'max_score': None,
            'fp_results': fp_results
        }

    all_tokens = list(tokens_and_fingerprints.keys())
    all_fingerprints = [tokens_and_fingerprints[key]['fingerprint'] for key in all_tokens]
    all_dates = [tokens_and_fingerprints[key]['date_added'] for key in all_tokens]
    # Find the closest token for this batch
    # Note: null fingerprint values are automatically handled and don't need to be filtered out.
    if data_type == 'audio':
        find_closest_func = find_closest_audio_fingerprint_from_list
    else:
        # Text, video, and image fingerprinting are done the same way,
        #  so they are all treated as the same here.
        find_closest_func = find_closest_image_fingerprint_from_list
    closest_token, closest_fingerprint, closest_date, score = find_closest_func(
        input_dict['target_fp'],
        all_fingerprints,
        all_tokens,
        all_dates,
        min_similarity=min_similarity
    )

    return {
        'closest': closest_token,
        'closest_fp': closest_fingerprint,
        'closest_date': closest_date,
        'max_score': score,
        'fp_results': fp_results
    }


def fingerprint_lookup_direct(fp_results, db_manager, equality_conditions=None):
    target_fingerprint = fp_results['result']
    token = fp_results['fp_token']

    if not fp_results['perform_lookup']:
        # The results are returned as a list to mimic the behavior of the group-chord fingerprint lookup path
        return [{
            'closest': None,
            'closest_fp': None,
            'closest_date': None,
            'max_score': None,
            'fp_results': fp_results
        }]

    if equality_conditions is None:
        equality_conditions = dict()
    equality_conditions['fingerprint'] = target_fingerprint

    tokens_and_fingerprints = db_manager.get_all_details(
        ['fingerprint', 'date_added'], start=0, limit=-1, exclude_token=token,
        allow_nulls=False, equality_conditions=equality_conditions
    )

    if tokens_and_fingerprints is None or len(tokens_and_fingerprints) == 0:
        return [{
            'closest': None,
            'closest_fp': None,
            'closest_date': None,
            'max_score': None,
            'fp_results': fp_results
        }]

    all_tokens = list(tokens_and_fingerprints.keys())
    all_fingerprints = [tokens_and_fingerprints[key]['fingerprint'] for key in all_tokens]
    all_dates = [tokens_and_fingerprints[key]['date_added'] for key in all_tokens]

    # Since the results are ordered by `date_added`, the first element is the earliest match
    closest_token, closest_fingerprint, closest_date, score = all_tokens[0], all_fingerprints[0], all_dates[0], 1

    return [{
        'closest': closest_token,
        'closest_fp': closest_fingerprint,
        'closest_date': closest_date,
        'max_score': score,
        'fp_results': fp_results
    }]


def fingerprint_lookup_callback(results_list, db_manager):
    """
    Handles the collection and aggregation of parallel fingerprint lookup results, plus database insertion.
    Args:
        results_list: List of parallel fingerprint lookup results
        db_manager: DBCachingManagerBase object

    Returns:
        Results of fingerprinting and the closest match
    """
    # Passing fingerprinting results along if it's been unsuccessful or a cached result has been returned
    # This is essentially the same check as in all the other find_closest tasks.
    fp_results = results_list[0]['fp_results']
    original_token = fp_results['fp_token']
    if not fp_results['perform_lookup']:
        if original_token is not None:
            closest = db_manager.get_closest_match(original_token)
        else:
            closest = None
        return {
            'closest': closest,
            'score': None,
            'fp_results': fp_results
        }
    results = [(x['closest'], x['closest_fp'], x['closest_date'], x['max_score']) for x in results_list]
    results = [x for x in results if x[0] is not None]
    # We sort the results by their dates (asc)
    results = sorted(results, key=lambda x: x[2])
    # If all results are null and the list is thus empty, then no closest fingerprint has been found,
    # and therefore, the closest token to this one is itself.
    if len(results) == 0:
        closest_token = original_token
        max_score = -1
    else:
        max_score = max([x[3] for x in results])
        # Since the results were sorted (asc) by date, the first one with the max score is the one with the lowest
        # date_added value. This ensures consistency in closest-token assignments because it ensures that even if there
        # are multiple matches, the same one will be chosen every time -- the one that was created first.
        closest_token = [x[0] for x in results if x[3] == max_score][0]
        closest_token = db_manager.get_closest_match(closest_token)
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
    # Ignoring the fingerprinting results and returning the results relevant to the task chain.
    # Used in tasks like transcription and OCR, where fingerprinting is performed before the task itself, but where
    # the results of the fingerprinting are not returned.
    return results_to_return
