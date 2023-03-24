from celery import shared_task, chain
from graphai.api.common.video import remove_silence_doublesided, perceptual_hash_audio, video_db_manager

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint', ignore_result=False)
def compute_audio_fingerprint_task(self, token, force=False, remove_silence=False, threshold=0.0):
    if remove_silence:
        fp_token, fresh, duration = remove_silence_doublesided(token, force=force, threshold=threshold)
        if fp_token is None:
            return {
                'result': None,
                'fresh': False,
                'duration': 0.0
            }
    else:
        fp_token = token
        fresh = True
    fingerprint, decoded, new_duration = perceptual_hash_audio(fp_token)
    return {
        'result': fingerprint,
        'fresh': fresh,
        'duration': new_duration
    }

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_callback', ignore_result=False)
def compute_audio_fingerprint_callback_task(self, results, token):
    if results['result'] is not None and results['fresh']:
        video_db_manager.insert_or_update_audio_details(
            token,
            {
                'fingerprint': results['result'],
            }
        )
    return results


def compute_audio_fingerprint_master(token, force=False, remove_silence=False, threshold=0.0):
    task = (compute_audio_fingerprint_task.s(token, force, remove_silence, threshold) |
                 compute_audio_fingerprint_callback_task.s(token)).\
            apply_async(priority=2)
    return {'task_id': task.id}