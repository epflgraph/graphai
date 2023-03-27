from celery import shared_task, chain, group
from graphai.api.common.video import remove_silence_doublesided, perceptual_hash_audio, \
    video_db_manager, video_config, find_closest_audio_fingerprint


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_silenceremoval', ignore_result=False)
def remove_audio_silence_task(self, token, force=False, threshold=0.0):
    input_filename_with_path = video_config.generate_filename(token)
    audio_type = token.split('.')[-1]
    output_suffix = '_nosilence.' + audio_type
    output_token = token + output_suffix
    output_filename_with_path = video_config.generate_filename(output_token)
    existing = video_db_manager.get_audio_details(token, cols=['nosilence_token', 'nosilence_duration'])
    if not force:
        if existing is not None and existing['nosilence_token'] is not None:
            print('Returning cached result')
            return {
                'fp_token': existing['nosilence_token'],
                'fresh': False,
                'duration': existing['nosilence_duration']
            }
    fp_token, duration = remove_silence_doublesided(input_filename_with_path, output_filename_with_path,
                                                    output_token, threshold=threshold)
    if fp_token is None:
        return {
            'fp_token': None,
            'fresh': False,
            'duration': 0.0
        }
    return {
        'fp_token': fp_token,
        'fresh': True,
        'duration': duration
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_silenceremoval_callback', ignore_result=False)
def remove_audio_silence_callback_task(self, result, audio_token):
    if result['fp_token'] is not None and result['fresh']:
        video_db_manager.insert_or_update_audio_details(
            audio_token,
            {
                'nosilence_token': result['fp_token'],
                'nosilence_duration': result['duration']
            }
        )
    return result


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint', ignore_result=False)
def compute_audio_fingerprint_task(self, input_dict, audio_token, force=False):
    fp_token = input_dict['fp_token']
    if fp_token is None:
        return {
            'result': None,
            'fresh': False,
            'duration': 0.0,
            'fp_nosilence': 0
        }
    existing = video_db_manager.get_audio_details(audio_token, cols=['fingerprint', 'duration',
                                                                     'nosilence_duration', 'fp_nosilence'])
    if not force:
        if existing is not None and existing['fingerprint'] is not None:
            print('Returning cached result')
            return {
                'result': existing['fingerprint'],
                'fresh': False,
                'duration': existing['duration'] if existing['fp_nosilence'] == 0 else existing['nosilence_duration'],
                'fp_nosilence': existing['fp_nosilence']
            }
    fp_token_with_path = video_config.generate_filename(fp_token)
    fingerprint, decoded = perceptual_hash_audio(fp_token_with_path)
    if fingerprint is None:
        return {
            'result': None,
            'fresh': False,
            'duration': 0.0,
            'fp_nosilence': 0
        }
    if input_dict.get('duration', None) is None:
        duration = existing['duration']
        fp_nosilence = 0
    else:
        duration = input_dict['duration']
        fp_nosilence = 1
    return {
        'result': fingerprint,
        'fresh': True,
        'duration': duration,
        'fp_nosilence': fp_nosilence
    }

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_callback', ignore_result=False)
def compute_audio_fingerprint_callback_task(self, results, token):
    if results['result'] is not None and results['fresh']:
        video_db_manager.insert_or_update_audio_details(
            token,
            {
                'fingerprint': results['result'],
                'fp_nosilence': results['fp_nosilence']
            }
        )
    return results


def resolve_most_similar_chain(token):
    if token is None:
        return None
    prev_most_similar = video_db_manager.get_closest_match_from_db(token)
    if prev_most_similar is None or prev_most_similar==token:
        return token
    return resolve_most_similar_chain(prev_most_similar)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def audio_fingerprint_find_closest_retrieve_from_db_task(self, results, token):
    target_fingerprint = results['result']
    fresh = results['fresh']
    if target_fingerprint is None or not fresh:
        return {
            'target_fp': None,
            'all_tokens': None,
            'all_fingerprints': None,
            'fp_results': results
        }
    tokens_and_fingerprints = video_db_manager.get_all_audio_details(['fingerprint'])
    print(tokens_and_fingerprints)
    all_tokens = list(tokens_and_fingerprints.keys())
    all_fingerprints = [tokens_and_fingerprints[key]['fingerprint'] for key in all_tokens]
    index_to_remove = all_tokens.index(token)
    del all_tokens[index_to_remove]
    del all_fingerprints[index_to_remove]
    return {
        'target_fp': target_fingerprint,
        'all_tokens': all_tokens,
        'all_fingerprints': all_fingerprints,
        'fp_results': results
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_find_closest_parallel', ignore_result=False)
def audio_fingerprint_find_closest_parallel_task(self, input_dict, n, n_total, min_similarity=0.8):
    if input_dict['all_tokens'] is None or len(input_dict['all_tokens']) == 0:
        return {
            'closest': None,
            'closest_fp': None,
            'max_score': None,
            'fp_results': input_dict['fp_results']
        }
    n_tokens_all = len(input_dict['all_tokens'])
    start_index = int(n/n_total*n_tokens_all)
    end_index = int((n+1)/n_total*n_tokens_all)
    closest_token, closest_fingerprint, score = find_closest_audio_fingerprint(
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


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_find_closest_callback', ignore_result=False)
def audio_fingerprint_find_closest_callback_task(self, results_list, original_token):
    fp_results = results_list[0]['fp_results']
    if fp_results['result'] is None:
        return{
            'closest': None,
            'score': None,
            'fp_results': fp_results
        }
    results = [(x['closest'], x['closest_fp'], x['max_score']) for x in results_list]
    results = [x for x in results if x[0] is not None]
    if len(results) == 0:
        closest_token = original_token
        max_score = -1
    else:
        max_score = max([x[2] for x in results])
        closest_token = [x[0] for x in results if x[2] == max_score][0]
        closest_token = resolve_most_similar_chain(closest_token)
    video_db_manager.insert_or_update_closest_match(
        original_token,
        {
            'most_similar_token': closest_token
        }
    )
    return {'closest': closest_token, 'score': max_score, 'fp_results': fp_results}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_fingerprint_task_callback', ignore_result=False)
def retrieve_fingerprint_callback_task(self, results):
    return results['fp_results']


def compute_audio_fingerprint_master(token, force=False, remove_silence=False, threshold=0.0,
                                     min_similarity=0.8, n_jobs=8):
    if not remove_silence:
        task = (compute_audio_fingerprint_task.s({'fp_token': token}, token, force) |
                compute_audio_fingerprint_callback_task.s(token) |
                audio_fingerprint_find_closest_retrieve_from_db_task.s(token) |
                group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)) |
                audio_fingerprint_find_closest_callback_task.s(token) |
                retrieve_fingerprint_callback_task.s()
                )
    else:
        task = (remove_audio_silence_task.s(token, force, threshold) |
                remove_audio_silence_callback_task.s(token) |
                compute_audio_fingerprint_task.s(token, force) |
                compute_audio_fingerprint_callback_task.s(token) |
                audio_fingerprint_find_closest_retrieve_from_db_task.s(token) |
                group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)) |
                audio_fingerprint_find_closest_callback_task.s(token) |
                retrieve_fingerprint_callback_task.s())
    task = task.apply_async(priority=2)
    return {'task_id': task.id}