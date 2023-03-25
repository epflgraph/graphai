from celery import shared_task, chain
from graphai.api.common.video import remove_silence_doublesided, perceptual_hash_audio, video_db_manager, video_config


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_silenceremoval', ignore_result=False)
def remove_audio_silence_task(self, token, force=False, threshold=0.0):
    fp_token, fresh, duration = remove_silence_doublesided(token, force=force, threshold=threshold)
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


def compute_audio_fingerprint_master(token, force=False, remove_silence=False, threshold=0.0):
    if not remove_silence:
        task = (compute_audio_fingerprint_task.s({'fp_token': token}, token, force) |
                 compute_audio_fingerprint_callback_task.s(token)).\
            apply_async(priority=2)
    else:
        task = (remove_audio_silence_task.s(token, force, threshold) |
                remove_audio_silence_callback_task.s(token) |
                compute_audio_fingerprint_task.s(token, force) |
                compute_audio_fingerprint_callback_task.s(token)). \
            apply_async(priority=2)
    return {'task_id': task.id}