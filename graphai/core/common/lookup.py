from graphai.core.common.fingerprinting import (
    find_closest_audio_fingerprint_from_list,
    find_closest_text_fingerprint_from_list,
    find_closest_image_fingerprint_from_list
)


def lookup_latest_allowed_date(fp_tokens, db_manager):
    if isinstance(fp_tokens, list):
        token_to_use_for_date_lookup = fp_tokens[0]
    else:
        token_to_use_for_date_lookup = fp_tokens
    # If we're here, the fp_token is not null
    latest_allowed_date = db_manager.get_details(
        token_to_use_for_date_lookup, ['date_added'], using_most_similar=False
    )[0]['date_added']
    return latest_allowed_date


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
    fp_tokens = results['fp_token']

    # If the fingerprint computation has been unsuccessful, we just pass the fingerprinting results along.
    if not results['perform_lookup']:
        return {
            'target_fp': None,
            'cache_count': None,
            'fp_results': results
        }

    ####################
    # Latest date lookup
    ####################
    latest_allowed_date = lookup_latest_allowed_date(fp_tokens, db_manager)

    # Retrieving all the tokens and their fingerprints. Since at least one audio has been extracted
    # (i.e. this one), this result is never null. In addition, there's at least one non-null fingerprint
    # value (again, for the present file).
    n_cache_rows = db_manager.get_cache_count(['fingerprint'], equality_conditions=equality_conditions)

    return {
        'target_fp': target_fingerprint,
        'cache_count': n_cache_rows,
        'fp_results': results,
        'latest_allowed_date': latest_allowed_date
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

    ##################
    # Unpacking values
    ##################
    fp_results = input_dict['fp_results']
    n_tokens_all = input_dict['cache_count']
    fp_tokens = fp_results['fp_token']
    target_fingerprints = input_dict['target_fp']
    latest_allowed_date = input_dict.get('latest_allowed_date', None)

    #############
    # Exit checks
    #############

    # Exit if lookup is disabled or if no previous relevant fingerprints exist
    if not fp_results['perform_lookup'] or n_tokens_all is None or n_tokens_all == 0:
        return {
            'closest': None,
            'closest_fp': None,
            'closest_date': None,
            'max_score': None,
            'fp_results': fp_results
        }

    # Compute the start and end indices
    start_index = int(i / n_total * n_tokens_all)
    end_index = int((i + 1) / n_total * n_tokens_all)
    limit = end_index - start_index

    # Exit if the length of the assigned segment is zero
    if limit <= 0:
        return {
            'closest': None,
            'closest_fp': None,
            'closest_date': None,
            'max_score': None,
            'fp_results': fp_results
        }

    ##############
    # Calculations
    ##############

    # Get all fingerprints with their tokens, excluding the tokens of the current computation
    tokens_and_fingerprints = db_manager.get_all_details(
        ['fingerprint', 'date_added'], start=start_index, limit=limit, exclude_token=fp_tokens,
        allow_nulls=False, equality_conditions=equality_conditions, latest_date=latest_allowed_date
    )

    # If there are no fingerprints in the cache at all, we exit
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
    elif data_type == 'text':
        find_closest_func = find_closest_text_fingerprint_from_list
    else:
        # Video and image fingerprinting are done the same way,
        # so they are both treated the same here.
        find_closest_func = find_closest_image_fingerprint_from_list

    closest_token, closest_fingerprint, closest_date, score = find_closest_func(
        target_fingerprints,
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
    ###########
    # Unpacking
    ###########
    target_fingerprints = fp_results['result']
    fp_tokens = fp_results['fp_token']
    # Cast to list
    has_been_cast_to_list = False
    if not isinstance(target_fingerprints, list):
        target_fingerprints = [target_fingerprints]
        has_been_cast_to_list = True
    # Initialize equality conditions if null
    if equality_conditions is None:
        equality_conditions = dict()

    ############
    # Exit check
    ############
    if not fp_results['perform_lookup']:
        # The results are returned as a list to mimic the behavior of the group-chord fingerprint lookup path
        return [{
            'closest': None,
            'closest_fp': None,
            'closest_date': None,
            'max_score': None,
            'fp_results': fp_results
        }]

    #############
    # Calculation
    #############
    closest_token = list()
    closest_fingerprint = list()
    best_date = list()
    score = list()
    latest_allowed_date = lookup_latest_allowed_date(fp_tokens, db_manager)
    for target_fingerprint in target_fingerprints:
        current_equality_conditions = equality_conditions.copy()
        current_equality_conditions['fingerprint'] = target_fingerprint

        tokens_and_fingerprints = db_manager.get_all_details(
            ['fingerprint', 'date_added'], start=0, limit=-1, exclude_token=fp_tokens,
            allow_nulls=False, equality_conditions=current_equality_conditions, latest_date=latest_allowed_date
        )

        if tokens_and_fingerprints is None or len(tokens_and_fingerprints) == 0:
            closest_token.append(None)
            closest_fingerprint.append(None)
            best_date.append(None)
            score.append(None)
        else:
            all_tokens = list(tokens_and_fingerprints.keys())
            all_fingerprints = [tokens_and_fingerprints[key]['fingerprint'] for key in all_tokens]
            all_dates = [tokens_and_fingerprints[key]['date_added'] for key in all_tokens]

            # Since the results are ordered by `date_added`, the first element is the earliest match
            current_closest_token, current_closest_fingerprint, current_best_date, current_score = (
                all_tokens[0], all_fingerprints[0], all_dates[0], 1
            )
            closest_token.append(current_closest_token)
            closest_fingerprint.append(current_closest_fingerprint)
            best_date.append(current_best_date)
            score.append(current_score)

    if has_been_cast_to_list:
        # If the input was a single fingerprint, we take the results out of their lists before returning them
        closest_token = closest_token[0]
        closest_fingerprint = closest_fingerprint[0]
        best_date = best_date[0]
        score = score[0]

    # The return value is a list of dictionaries (well, one dictionary) to mimic the behavior of a group + chord
    return [{
        'closest': closest_token,
        'closest_fp': closest_fingerprint,
        'closest_date': best_date,
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
    ###########
    # Unpacking
    ###########
    fp_results = results_list[0]['fp_results']
    original_tokens = fp_results['fp_token']
    results = [(x.get('closest', None),
                x.get('closest_fp', None),
                x.get('closest_date', None),
                x.get('max_score', None))
               for x in results_list]
    has_been_cast_to_list = False
    if original_tokens is not None and not isinstance(original_tokens, list):
        original_tokens = [original_tokens]
        results = [([x[0]], [x[1]], [x[2]], [x[3]]) for x in results]
        has_been_cast_to_list = True

    ############
    # Exit check
    ############
    if not fp_results['perform_lookup']:
        if original_tokens is not None:
            closest = [db_manager.get_closest_match(original_token) for original_token in original_tokens]
            if has_been_cast_to_list:
                closest = closest[0]
        else:
            closest = None
        return {
            'closest': closest,
            'score': None,
            'fp_results': fp_results
        }

    #############
    # Calculation
    #############
    closest_token = list()
    max_score = list()
    for i in range(len(original_tokens)):
        current_original_token = original_tokens[i]
        current_results = [(x[0][i], x[1][i], x[2][i], x[3][i]) for x in results]
        current_results = [x for x in current_results if x[0] is not None]
        # We sort the results by their dates (asc)
        current_results = sorted(current_results, key=lambda x: x[2])
        # If all results are null and the list is thus empty, then no closest fingerprint has been found,
        # and therefore, the closest token to this one is itself.
        if len(current_results) == 0:
            current_closest_token = current_original_token
            current_max_score = -1
        else:
            current_max_score = max([x[3] for x in current_results])
            # Since the results were sorted (asc) by date, the first one with the max score is the one with the lowest
            # date_added value. This ensures consistency in closest-token assignments because it ensures that even if there
            # are multiple matches, the same one will be chosen every time -- the one that was created first.
            current_closest_token = [x[0] for x in current_results if x[3] == current_max_score][0]
            current_closest_token = db_manager.get_closest_match(current_closest_token)
        # Whether the closest token is itself or another token, we store the result in the database.
        print(current_closest_token)
        db_manager.insert_or_update_closest_match(
            current_original_token,
            {
                'most_similar_token': current_closest_token
            }
        )
        closest_token.append(current_closest_token)
        max_score.append(current_max_score)
    if has_been_cast_to_list:
        closest_token = closest_token[0]
        max_score = max_score[0]
    return {'closest': closest_token, 'score': max_score, 'fp_results': fp_results}


def retrieve_fingerprint_callback(results, db_manager, has_origin=True):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    results_to_return = results['fp_results']
    results_to_return['closest'] = results['closest']
    if has_origin:
        if results_to_return['closest'] is not None:
            results_to_return['closest_origin'] = db_manager.get_origin(results_to_return['closest'])
        else:
            results_to_return['closest_origin'] = None
    return results_to_return


def add_token_status_to_results(results, token_status_func):
    results['token_status'] = token_status_func(results['token'])
    return results


def ignore_fingerprint_results_callback(results, token_status_func):
    # Ignoring the fingerprinting results and returning the results relevant to the task chain.
    # Used in tasks like transcription and OCR, where fingerprinting is performed before the task itself, but where
    # the results of the fingerprinting are not returned.
    results_to_return = results['fp_results']['original_results']
    return add_token_status_to_results(results_to_return, token_status_func)


def is_fingerprinted(token, db_manager):
    values = db_manager.get_details(token, ['fingerprint'], using_most_similar=False)[0]
    exists = values is not None
    return exists, exists and values['fingerprint'] is not None


def fingerprint_cache_lookup(token, db_manager):
    existing = db_manager.get_details(token, cols=['fingerprint'])[0]
    if existing is not None and existing['fingerprint'] is not None:
        existing_closest = db_manager.get_closest_match(token)
        return {
            'result': existing['fingerprint'],
            'closest': existing_closest,
            'fp_token': existing['id_token'],
            'perform_lookup': False,
            'fresh': False
        }
    return None


def fingerprint_cache_lookup_with_most_similar(token, db_manager, extra_cols=None):
    cols = ['fingerprint']
    if extra_cols is not None:
        cols = cols + extra_cols
    existing_list = db_manager.get_details(token, cols=cols,
                                           using_most_similar=True)
    for existing in existing_list:
        if existing is None:
            continue
        if existing['fingerprint'] is not None:
            # We have a cache hit, now we gather all the results that should be returned
            # The closest match
            existing_closest = db_manager.get_closest_match(token)
            if existing_closest is not None:
                # If the closest match exists, we also want to return its origin token
                existing_closest_origin = db_manager.get_origin(existing_closest)
            else:
                existing_closest_origin = None
            print('Returning cached result')
            final_result = {
                'result': existing['fingerprint'],
                'fresh': False,
                'closest': existing_closest,
                'closest_origin': existing_closest_origin
            }
            if extra_cols is not None:
                for extra_col in extra_cols:
                    final_result[extra_col] = existing[extra_col]
            return final_result

    return None


def cache_lookup_generic(token, db_manager, cols):
    existing_list = db_manager.get_details(token, cols,
                                           using_most_similar=True)
    for existing in existing_list:
        if existing is None:
            continue
        if all(existing[key] is not None for key in cols):
            print('Returning cached result')
            result = {
                'token': token,
                'fresh': False
            }
            for col in cols:
                result[col] = existing[col]
            return result
    return None


def database_callback_generic(token, db_manager, values_dict, force=False, use_closest_match=False):
    db_manager.insert_or_update_details(
        token, values_dict
    )
    if use_closest_match and not force:
        closest = db_manager.get_closest_match(token)
        if closest is not None and closest != token:
            db_manager.insert_or_update_details(
                closest, values_dict
            )
            return closest
    return token


def token_based_text_lookup(token, db_manager, main_col, modify_result_func=None, modify_result_args=None, **kwargs):
    existing = db_manager.get_details(token, [main_col], using_most_similar=False)[0]
    if existing is not None and existing[main_col] is not None:
        print('Returning cached result')
        final_result = {
            'result': existing[main_col],
            'successful': True,
            'text_too_large': False,
            'fresh': False,
            'device': None
        }
        if modify_result_func is not None and modify_result_args is not None:
            final_result['result'] = modify_result_func(final_result['result'], **modify_result_args)
        if len(kwargs) > 0:
            for key in kwargs:
                final_result[key] = kwargs[key]
        return final_result
    return None


def fingerprint_based_text_lookup(token, fp, db_manager, main_col, extra_cols, equality_conditions,
                                  modify_result_func=None, modify_result_args=None,
                                  **kwargs):
    equality_conditions['fingerprint'] = fp
    closest_match = db_manager.get_all_details([main_col] + extra_cols, allow_nulls=False,
                                               equality_conditions=equality_conditions)
    if closest_match is not None:
        all_keys = list(closest_match.keys())
        cached_result = closest_match[all_keys[0]][main_col]
        db_manager.insert_or_update_details(token, {
            main_col: cached_result
        })
        final_result = {
            'result': cached_result,
            'successful': True,
            'text_too_large': False,
            'fresh': False,
            'device': None
        }
        if modify_result_func is not None and modify_result_args is not None:
            final_result['result'] = modify_result_func(final_result['result'], **modify_result_args)
        if len(kwargs) > 0:
            for key in kwargs:
                final_result[key] = kwargs[key]
        return final_result
    return None
