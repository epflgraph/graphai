from celery import shared_task

from graphai.api.common.ontology import ontology_data
from graphai.api.common.translation import translation_models
from graphai.api.common.video import (
    transcription_model,
    local_ocr_nlp_models
)
from graphai.core.common.common_utils import strtobool
from graphai.core.common.config import config

from graphai.core.common.video import (
    find_closest_audio_fingerprint_from_list,
    find_closest_image_fingerprint_from_list,
    find_closest_text_fingerprint_from_list
)
from graphai.core.common.text_utils import perceptual_hash_text


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


def compute_text_fingerprint_common(token, text):
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

    # If the fingerprint computation has been unsuccessful, we just pass the fingerprinting results along.
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

    ##################
    # Unpacking values
    ##################
    fp_results = input_dict['fp_results']
    n_tokens_all = input_dict['cache_count']
    fp_tokens = fp_results['fp_token']
    target_fingerprints = input_dict['target_fp']

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
        allow_nulls=False, equality_conditions=equality_conditions
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
        # Text, video, and image fingerprinting are done the same way,
        #  so they are all treated as the same here.
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
    for target_fingerprint in target_fingerprints:
        current_equality_conditions = equality_conditions.copy()
        current_equality_conditions['fingerprint'] = target_fingerprint

        tokens_and_fingerprints = db_manager.get_all_details(
            ['fingerprint', 'date_added'], start=0, limit=-1, exclude_token=fp_tokens,
            allow_nulls=False, equality_conditions=equality_conditions
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
def ignore_fingerprint_results_callback_task(self, results, results_to_return=None):
    # Ignoring the fingerprinting results and returning the results relevant to the task chain.
    # Used in tasks like transcription and OCR, where fingerprinting is performed before the task itself, but where
    # the results of the fingerprinting are not returned.
    if results_to_return is None:
        return results['fp_results']['original_results']
    else:
        return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.text_dummy_task', ignore_result=False)
def text_dummy_task(self, results):
    # This task is required for chaining groups due to the peculiarities of celery
    # Whenever there are two groups in one chain of tasks, there need to be at least
    # TWO tasks between them, and this dummy task is simply an f(x)=x function.
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.video_dummy_task', ignore_result=False)
def video_dummy_task(self, results):
    # This task is required for chaining groups due to the peculiarities of celery
    # Whenever there are two groups in one chain of tasks, there need to be at least
    # TWO tasks between them, and this dummy task is simply an f(x)=x function.
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.init', ignore_result=False,
             transcription_obj=transcription_model,
             nlp_obj=local_ocr_nlp_models,
             translation_obj=translation_models,
             ontology_data_obj=ontology_data)
def video_init_task(self):
    # This task initialises the video celery worker by loading into memory the transcription and NLP models
    print('Start video_init task')

    if strtobool(config['preload']['video']):
        print('Loading transcription model...')
        self.transcription_obj.load_model_whisper()

        print('Loading NLP models...')
        self.nlp_obj.load_nlp_models()

        print('Loading translation models...')
        self.translation_obj.load_models()
    else:
        print('Skipping preloading for video endpoints.')

    if strtobool(config['preload']['ontology']):
        print('Loading ontology data...')
        self.ontology_data_obj.load_data()
    else:
        print('Skipping preloading for ontology endpoints.')

    print('All video processing objects loaded')
    return True
